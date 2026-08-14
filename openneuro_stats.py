#!/usr/bin/env python3
"""Compute comprehensive statistics for OpenNeuro EEG datasets in BIDS format."""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path


EEG_FORMAT_EXTENSIONS = {
    ".set": "EEGLAB (.set)",
    ".fdt": "EEGLAB (.fdt)",
    ".vhdr": "BrainVision (.vhdr)",
    ".eeg": "BrainVision (.eeg)",
    ".vmrk": "BrainVision (.vmrk)",
    ".bdf": "BioSemi (.bdf)",
    ".edf": "EDF (.edf)",
    ".fif": "MNE/Elekta (.fif)",
    ".gz": "Compressed (.gz)",
}

EEG_PRIMARY_FORMATS = {".set", ".vhdr", ".bdf", ".edf", ".fif"}


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def parse_dataset_description(ds_path):
    desc_file = ds_path / "dataset_description.json"
    if not desc_file.exists():
        return {}
    with open(desc_file) as f:
        data = json.load(f)
    return {
        "name": data.get("Name", ""),
        "license": data.get("License", ""),
        "bids_version": data.get("BIDSVersion", ""),
        "authors": data.get("Authors", []),
        "doi": data.get("DatasetDOI", ""),
    }


def parse_participants(ds_path):
    tsv_file = ds_path / "participants.tsv"
    if not tsv_file.exists():
        return {"count": 0, "ages": [], "sex_counts": Counter()}

    with open(tsv_file, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        return {"count": 0, "ages": [], "sex_counts": Counter()}

    headers_lower = {h.lower(): h for h in rows[0].keys()}

    age_key = None
    for candidate in ("age", "Age"):
        if candidate in rows[0]:
            age_key = candidate
            break
    if age_key is None and "age" in headers_lower:
        age_key = headers_lower["age"]

    sex_key = None
    for candidate in ("sex", "gender", "Gender", "Sex"):
        if candidate in rows[0]:
            sex_key = candidate
            break
    if sex_key is None:
        for k in ("sex", "gender"):
            if k in headers_lower:
                sex_key = headers_lower[k]
                break

    ages = []
    sex_counts = Counter()

    for row in rows:
        if age_key and age_key in row:
            val = row[age_key].strip()
            if val and val.lower() not in ("n/a", "na", "nan", ""):
                try:
                    ages.append(float(val))
                except ValueError:
                    pass

        if sex_key and sex_key in row:
            val = row[sex_key].strip().upper()
            if val and val not in ("N/A", "NA", "NAN", ""):
                if val in ("M", "MALE"):
                    sex_counts["M"] += 1
                elif val in ("F", "FEMALE"):
                    sex_counts["F"] += 1
                else:
                    sex_counts["Other"] += 1

    return {"count": len(rows), "ages": ages, "sex_counts": sex_counts}


def parse_eeg_channels_from_tsv(ds_path):
    """Extract EEG channel names and count from channels.tsv files.

    Returns the most common EEG channel name list across all recordings.
    If no EEG-typed channels exist, falls back to MISC-typed channels
    (some datasets mislabel EEG channels as MISC).
    """
    eeg_lists = Counter()
    misc_lists = Counter()
    all_lists = Counter()
    for tsv_path in ds_path.rglob("*_channels.tsv"):
        try:
            with open(tsv_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter="\t")
                headers_lower = {h.lower(): h for h in (reader.fieldnames or [])}
                type_key = headers_lower.get("type")
                name_key = headers_lower.get("name")
                if name_key is None:
                    continue
                eeg_names = []
                misc_names = []
                all_names = []
                for row in reader:
                    ch_type = row.get(type_key, "").strip().upper() if type_key else ""
                    ch_name = row.get(name_key, "").strip()
                    all_names.append(ch_name)
                    if ch_type == "EEG":
                        eeg_names.append(ch_name)
                    elif ch_type == "MISC":
                        misc_names.append(ch_name)
            if eeg_names:
                eeg_lists[tuple(eeg_names)] += 1
            if misc_names:
                misc_lists[tuple(misc_names)] += 1
            if all_names:
                all_lists[tuple(all_names)] += 1
        except OSError:
            pass
    if eeg_lists:
        return list(eeg_lists.most_common(1)[0][0])
    if misc_lists:
        return list(misc_lists.most_common(1)[0][0])
    if all_lists:
        return list(all_lists.most_common(1)[0][0])
    return None


def parse_eeg_channels_from_vhdr(ds_path):
    """Fallback: parse channel names from BrainVision .vhdr header files."""
    for vhdr_path in ds_path.rglob("*.vhdr"):
        try:
            with open(vhdr_path, encoding="utf-8", errors="ignore") as f:
                in_section = False
                names = []
                for line in f:
                    line = line.strip()
                    if line == "[Channel Infos]":
                        in_section = True
                        continue
                    if in_section:
                        if line.startswith("["):
                            break
                        if line.startswith("Ch") and "=" in line:
                            parts = line.split("=", 1)[1].split(",")
                            if parts:
                                names.append(parts[0].strip())
                if names:
                    return names
        except OSError:
            pass
    return None


NON_EEG_LABELS = {
    "ECG", "EKG", "EMG", "EOG", "VEOG", "HEOG",
    "EDF ANNOTATIONS", "STATUS", "STI 014", "TRIGGER",
    "GSR", "GSR1", "GSR2", "RESP", "PLET", "TEMP",
    "EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8",
    "ERG1", "ERG2",
}


def parse_eeg_channels_from_edf(ds_path):
    """Fallback: parse channel names from EDF/BDF header files."""
    for edf_path in ds_path.rglob("*.edf"):
        try:
            with open(edf_path, "rb") as f:
                header = f.read(256)
                n_signals = int(header[252:256].strip())
                labels_raw = f.read(16 * n_signals)
                labels = [
                    labels_raw[i * 16:(i + 1) * 16].decode("ascii", errors="ignore").strip()
                    for i in range(n_signals)
                ]
            eeg_labels = [l for l in labels if l.upper() not in NON_EEG_LABELS]
            if eeg_labels:
                return eeg_labels
        except (OSError, ValueError):
            pass
    return None


def get_eeg_channels(ds_path):
    """Get EEG channel list: channels.tsv -> .vhdr header -> .edf header."""
    channels = parse_eeg_channels_from_tsv(ds_path)
    if channels:
        return channels
    channels = parse_eeg_channels_from_vhdr(ds_path)
    if channels:
        return channels
    channels = parse_eeg_channels_from_edf(ds_path)
    if channels:
        return channels
    return []


def parse_eeg_params(ds_path):
    sampling_rates = set()
    durations = []
    references = set()
    manufacturers = set()

    for json_path in ds_path.rglob("*_eeg.json"):
        with open(json_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        if "SamplingFrequency" in data:
            sampling_rates.add(data["SamplingFrequency"])
        if "RecordingDuration" in data:
            try:
                durations.append(float(data["RecordingDuration"]))
            except (ValueError, TypeError):
                pass
        if "EEGReference" in data:
            ref = str(data["EEGReference"]).strip()
            if ref and ref.lower() not in ("n/a", "na"):
                references.add(ref)
        if "Manufacturer" in data:
            mfr = str(data["Manufacturer"]).strip()
            if mfr and mfr.lower() not in ("n/a", "na"):
                manufacturers.add(mfr)

    return {
        "sampling_rates": sorted(sampling_rates),
        "durations": durations,
        "references": sorted(references),
        "manufacturers": sorted(manufacturers),
    }


def count_files_and_formats(ds_path):
    ext_counts = Counter()
    total_size = 0
    eeg_formats = set()
    task_sizes = {}
    task_file_counts = {}
    task_subjects = {}
    ds_str = str(ds_path) + os.sep

    for root, _dirs, files in os.walk(ds_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            ext_counts[ext] += 1
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                fsize = 0
            total_size += fsize
            if ext in EEG_PRIMARY_FORMATS:
                eeg_formats.add(EEG_FORMAT_EXTENSIONS.get(ext, ext))

            task_name = _extract_task_from_filename(fname)
            if task_name:
                task_sizes[task_name] = task_sizes.get(task_name, 0) + fsize
                task_file_counts[task_name] = task_file_counts.get(task_name, 0) + 1
                rel = root[len(ds_str):]
                sub = rel.split(os.sep)[0] if rel else ""
                if sub.startswith("sub-"):
                    if task_name not in task_subjects:
                        task_subjects[task_name] = set()
                    task_subjects[task_name].add(sub)

    return {
        "ext_counts": ext_counts,
        "total_size": total_size,
        "eeg_formats": sorted(eeg_formats),
        "task_sizes": task_sizes,
        "task_file_counts": task_file_counts,
        "task_subjects": {t: len(s) for t, s in task_subjects.items()},
    }


def _extract_task_from_filename(fname):
    for part in fname.split("_"):
        if part.startswith("task-"):
            return part[5:]
    return None


def count_subjects_sessions(ds_path):
    subjects = sorted(
        d.name for d in ds_path.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    sessions = set()
    for sub_dir in (ds_path / s for s in subjects):
        for child in sub_dir.iterdir():
            if child.is_dir() and child.name.startswith("ses-"):
                sessions.add(child.name)
    return len(subjects), sorted(sessions)


def collect_dataset_stats(ds_path):
    ds_id = ds_path.name
    desc = parse_dataset_description(ds_path)
    participants = parse_participants(ds_path)
    eeg_params = parse_eeg_params(ds_path)
    files = count_files_and_formats(ds_path)
    n_subjects, sessions = count_subjects_sessions(ds_path)
    eeg_channels = get_eeg_channels(ds_path)

    subject_count = max(n_subjects, participants["count"])

    return {
        "id": ds_id,
        "name": desc.get("name", ""),
        "license": desc.get("license", ""),
        "bids_version": desc.get("bids_version", ""),
        "n_subjects": subject_count,
        "n_sessions": len(sessions),
        "sessions": sessions,
        "ages": participants["ages"],
        "sex_counts": participants["sex_counts"],
        "sampling_rates": eeg_params["sampling_rates"],
        "channel_count": len(eeg_channels) if eeg_channels else None,
        "eeg_channels": eeg_channels,
        "durations": eeg_params["durations"],
        "tasks": sorted(files["task_sizes"].keys()),
        "references": eeg_params["references"],
        "manufacturers": eeg_params["manufacturers"],
        "eeg_formats": files["eeg_formats"],
        "ext_counts": files["ext_counts"],
        "total_size": files["total_size"],
        "n_files": sum(files["ext_counts"].values()),
        "task_sizes": files["task_sizes"],
        "task_file_counts": files["task_file_counts"],
        "task_subjects": files["task_subjects"],
    }


def aggregate_stats(all_datasets):
    total_subjects = sum(d["n_subjects"] for d in all_datasets)
    total_files = sum(d["n_files"] for d in all_datasets)
    total_size = sum(d["total_size"] for d in all_datasets)

    all_ages = []
    all_sex = Counter()
    all_sampling_rates = Counter()
    all_channel_counts = Counter()
    all_durations = []
    all_tasks = set()
    all_references = Counter()
    all_manufacturers = Counter()
    all_formats = Counter()
    all_ext_counts = Counter()

    for d in all_datasets:
        all_ages.extend(d["ages"])
        all_sex += d["sex_counts"]
        for sr in d["sampling_rates"]:
            all_sampling_rates[sr] += 1
        if d["channel_count"] is not None:
            all_channel_counts[d["channel_count"]] += 1
        all_durations.extend(d["durations"])
        all_tasks.update(d["tasks"])
        for r in d["references"]:
            all_references[r] += 1
        for m in d["manufacturers"]:
            all_manufacturers[m] += 1
        for fmt in d["eeg_formats"]:
            all_formats[fmt] += 1
        all_ext_counts += d["ext_counts"]

    age_stats = {}
    if all_ages:
        age_stats = {
            "min": min(all_ages),
            "max": max(all_ages),
            "mean": statistics.mean(all_ages),
            "median": statistics.median(all_ages),
            "stdev": statistics.stdev(all_ages) if len(all_ages) > 1 else 0,
            "count": len(all_ages),
        }

    duration_stats = {}
    if all_durations:
        duration_stats = {
            "min": min(all_durations),
            "max": max(all_durations),
            "mean": statistics.mean(all_durations),
            "median": statistics.median(all_durations),
        }

    return {
        "n_datasets": len(all_datasets),
        "total_subjects": total_subjects,
        "total_files": total_files,
        "total_size": total_size,
        "age_stats": age_stats,
        "sex_counts": all_sex,
        "sampling_rates": all_sampling_rates,
        "channel_counts": all_channel_counts,
        "duration_stats": duration_stats,
        "n_tasks": len(all_tasks),
        "tasks": sorted(all_tasks),
        "references": all_references,
        "manufacturers": all_manufacturers,
        "eeg_formats": all_formats,
        "ext_counts": all_ext_counts,
    }


def print_report(all_datasets, agg):
    sep = "=" * 80
    subsep = "-" * 80

    print(sep)
    print("  OPENNEURO EEG DATASET STATISTICS")
    print(sep)

    print(f"\n  Datasets:       {agg['n_datasets']}")
    print(f"  Total subjects: {agg['total_subjects']}")
    print(f"  Total files:    {agg['total_files']:,}")
    print(f"  Total size:     {human_size(agg['total_size'])}")
    print(f"  Unique tasks:   {agg['n_tasks']}")

    # Per-dataset table
    print(f"\n{subsep}")
    print("  PER-DATASET SUMMARY")
    print(subsep)
    header = f"  {'ID':<12} {'Name':<40} {'Subj':>5} {'Ch':>6} {'SR (Hz)':>10} {'Files':>7} {'Size':>10} {'Format':<25}"
    print(header)
    print(f"  {'-'*12} {'-'*40} {'-'*5} {'-'*6} {'-'*10} {'-'*7} {'-'*10} {'-'*25}")
    for d in sorted(all_datasets, key=lambda x: x["id"]):
        name = d["name"][:39] if d["name"] else ""
        fmt = ", ".join(d["eeg_formats"])[:24] if d["eeg_formats"] else ""
        ch_str = str(d["channel_count"]) if d["channel_count"] is not None else ""
        sr = d["sampling_rates"]
        if not sr:
            sr_str = ""
        elif len(sr) <= 2:
            sr_str = "/".join(str(int(s) if s == int(s) else s) for s in sr)
        else:
            sr_str = f"{int(sr[0])}-{int(sr[-1])}"
        print(
            f"  {d['id']:<12} {name:<40} {d['n_subjects']:>5} "
            f"{ch_str:>6} {sr_str:>10} {d['n_files']:>7} "
            f"{human_size(d['total_size']):>10} {fmt:<25}"
        )

    # Demographics
    print(f"\n{subsep}")
    print("  DEMOGRAPHICS")
    print(subsep)
    if agg["age_stats"]:
        a = agg["age_stats"]
        print(f"  Age (n={a['count']:,} subjects with age data):")
        print(f"    Range:  {a['min']:.1f} - {a['max']:.1f}")
        print(f"    Mean:   {a['mean']:.1f}")
        print(f"    Median: {a['median']:.1f}")
        print(f"    Stdev:  {a['stdev']:.1f}")
    else:
        print("  No age data available.")

    print(f"\n  Sex/Gender (n={sum(agg['sex_counts'].values()):,}):")
    for label in ("M", "F", "Other"):
        count = agg["sex_counts"].get(label, 0)
        total = sum(agg["sex_counts"].values())
        pct = (count / total * 100) if total else 0
        print(f"    {label:<8} {count:>6}  ({pct:.1f}%)")

    # Recording parameters
    print(f"\n{subsep}")
    print("  RECORDING PARAMETERS")
    print(subsep)

    print("\n  Sampling frequencies (Hz) — datasets using each:")
    for sr, count in sorted(agg["sampling_rates"].items()):
        print(f"    {sr:>8} Hz : {count} dataset(s)")

    print(f"\n  Total channel counts (from channels.tsv) — datasets using each:")
    for cc, count in sorted(agg["channel_counts"].items()):
        print(f"    {cc:>4} ch : {count} dataset(s)")

    if agg["duration_stats"]:
        dur = agg["duration_stats"]
        print(f"\n  Recording duration (seconds, n={len([d for d in all_datasets if d['durations']])} datasets with data):")
        print(f"    Range:  {dur['min']:.1f} - {dur['max']:.1f}")
        print(f"    Mean:   {dur['mean']:.1f}")
        print(f"    Median: {dur['median']:.1f}")

    print("\n  EEG formats — datasets using each:")
    for fmt, count in agg["eeg_formats"].most_common():
        print(f"    {fmt:<25} : {count} dataset(s)")

    print("\n  EEG references — datasets using each:")
    for ref, count in agg["references"].most_common():
        print(f"    {ref:<30} : {count} dataset(s)")

    print("\n  Manufacturers — datasets using each:")
    for mfr, count in agg["manufacturers"].most_common():
        print(f"    {mfr:<35} : {count} dataset(s)")

    # File types
    print(f"\n{subsep}")
    print("  FILE TYPE BREAKDOWN")
    print(subsep)
    for ext, count in agg["ext_counts"].most_common(20):
        ext_label = ext if ext else "(no ext)"
        print(f"    {ext_label:<15} : {count:>8,}")

    # EEG channel lists
    print(f"\n{subsep}")
    print("  EEG CHANNEL LISTS")
    print(subsep)

    montage_groups = {}
    for d in sorted(all_datasets, key=lambda x: x["id"]):
        ch_key = tuple(d["eeg_channels"]) if d["eeg_channels"] else ()
        if ch_key:
            montage_groups.setdefault(ch_key, []).append(d["id"])

    for d in sorted(all_datasets, key=lambda x: x["id"]):
        if not d["eeg_channels"]:
            print(f"\n  {d['id']}  ({d['name'][:50]})")
            print(f"    (no EEG channel info)")
            continue
        ch_key = tuple(d["eeg_channels"])
        shared = montage_groups[ch_key]
        print(f"\n  {d['id']}  ({d['name'][:50]})")
        print(f"    {len(d['eeg_channels'])} EEG channels: {', '.join(d['eeg_channels'])}")
        if len(shared) > 1:
            others = [s for s in shared if s != d["id"]]
            print(f"    ** Same montage as: {', '.join(others)}")

    # Tasks
    print(f"\n{subsep}")
    print(f"  TASKS ({agg['n_tasks']} unique)")
    print(subsep)
    for task in agg["tasks"]:
        print(f"    {task}")

    print(f"\n{sep}")
    print()


def save_csv(all_datasets, output_path):
    fieldnames = [
        "id", "name", "task", "license", "bids_version",
        "n_subjects",
        "age_min", "age_max", "age_mean", "age_median",
        "n_male", "n_female", "n_other_sex",
        "sampling_rates", "channel_count", "eeg_channels",
        "eeg_formats", "references", "manufacturers",
        "n_files", "total_size_bytes", "total_size_human",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in sorted(all_datasets, key=lambda x: x["id"]):
            ages = d["ages"]
            base = {
                "id": d["id"],
                "name": d["name"],
                "license": d["license"],
                "bids_version": d["bids_version"],
                "n_subjects": d["n_subjects"],
                "age_min": f"{min(ages):.1f}" if ages else "",
                "age_max": f"{max(ages):.1f}" if ages else "",
                "age_mean": f"{statistics.mean(ages):.1f}" if ages else "",
                "age_median": f"{statistics.median(ages):.1f}" if ages else "",
                "n_male": d["sex_counts"].get("M", 0),
                "n_female": d["sex_counts"].get("F", 0),
                "n_other_sex": d["sex_counts"].get("Other", 0),
                "sampling_rates": "; ".join(str(s) for s in d["sampling_rates"]),
                "channel_count": d["channel_count"] if d["channel_count"] is not None else "",
                "eeg_channels": "; ".join(d["eeg_channels"]),
                "eeg_formats": "; ".join(d["eeg_formats"]),
                "references": "; ".join(d["references"]),
                "manufacturers": "; ".join(d["manufacturers"]),
                "n_files": d["n_files"],
                "total_size_bytes": d["total_size"],
                "total_size_human": human_size(d["total_size"]),
            }
            tasks = d["tasks"] if d["tasks"] else [""]
            for task in tasks:
                row = {
                    **base,
                    "task": task,
                    "n_subjects": d["task_subjects"].get(task, d["n_subjects"]) if task else d["n_subjects"],
                    "n_files": d["task_file_counts"].get(task, 0) if task else d["n_files"],
                    "total_size_bytes": d["task_sizes"].get(task, 0) if task else d["total_size"],
                    "total_size_human": human_size(d["task_sizes"].get(task, 0)) if task else human_size(d["total_size"]),
                }
                writer.writerow(row)
    print(f"CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistics for OpenNeuro EEG datasets in BIDS format.",
    )
    parser.add_argument(
        "--data-dir",
        default="/data/reve_public_dataset/openneuro",
        help="Path to the OpenNeuro datasets directory (default: /data/reve_public_dataset/openneuro).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save per-dataset statistics as CSV.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    ds_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("ds")
    )

    if not ds_dirs:
        print(f"No dataset directories found in {data_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(ds_dirs)} datasets in {data_dir} ...")

    all_datasets = []
    for i, ds_path in enumerate(ds_dirs, 1):
        print(f"  [{i}/{len(ds_dirs)}] {ds_path.name} ...", end="", flush=True)
        stats = collect_dataset_stats(ds_path)
        all_datasets.append(stats)
        print(f" {stats['n_subjects']} subjects, {stats['n_files']:,} files")

    agg = aggregate_stats(all_datasets)
    print_report(all_datasets, agg)

    if args.output_csv:
        save_csv(all_datasets, args.output_csv)


if __name__ == "__main__":
    main()
