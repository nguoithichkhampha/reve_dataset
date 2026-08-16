"""BIDS metadata parsers — pure data-in, data-out (no filesystem calls).

Refactored from openneuro_stats.py so the same logic works with GCS blobs.
"""

import csv
import io
import json
import statistics
from collections import Counter


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

NON_EEG_LABELS = {
    "ECG", "EKG", "EMG", "EOG", "VEOG", "HEOG",
    "EDF ANNOTATIONS", "STATUS", "STI 014", "TRIGGER",
    "GSR", "GSR1", "GSR2", "RESP", "PLET", "TEMP",
    "EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8",
    "ERG1", "ERG2",
}


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def parse_dataset_description(json_text):
    """Parse dataset_description.json content."""
    data = json.loads(json_text)
    return {
        "name": data.get("Name", ""),
        "license": data.get("License", ""),
        "bids_version": data.get("BIDSVersion", ""),
        "authors": data.get("Authors", []),
        "doi": data.get("DatasetDOI", ""),
    }


def parse_participants(tsv_text):
    """Parse participants.tsv content."""
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
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


def parse_channels_tsv(tsv_text):
    """Parse a single *_channels.tsv file.

    Returns (eeg_names, misc_names, all_names) lists.
    """
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    headers_lower = {h.lower(): h for h in (reader.fieldnames or [])}
    type_key = headers_lower.get("type")
    name_key = headers_lower.get("name")
    if name_key is None:
        return [], [], []

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

    return eeg_names, misc_names, all_names


def select_best_channel_list(channel_lists):
    """From a list of (eeg, misc, all) tuples, pick the best channel list.

    Priority: EEG-typed -> MISC-typed -> all channels.
    Returns the most common list as a plain list, or None.
    """
    eeg_counter = Counter()
    misc_counter = Counter()
    all_counter = Counter()

    for eeg_names, misc_names, all_names in channel_lists:
        if eeg_names:
            eeg_counter[tuple(eeg_names)] += 1
        if misc_names:
            misc_counter[tuple(misc_names)] += 1
        if all_names:
            all_counter[tuple(all_names)] += 1

    if eeg_counter:
        return list(eeg_counter.most_common(1)[0][0])
    if misc_counter:
        return list(misc_counter.most_common(1)[0][0])
    if all_counter:
        return list(all_counter.most_common(1)[0][0])
    return None


def parse_vhdr_channels(text):
    """Parse channel names from BrainVision .vhdr header text."""
    in_section = False
    names = []
    for line in text.splitlines():
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
    return names if names else None


def parse_edf_header_channels(header_bytes):
    """Parse channel names from EDF binary header bytes.

    Expects at least 256 bytes (fixed header), reads n_signals from bytes
    252-256, then needs 16 * n_signals additional bytes for labels.
    """
    if len(header_bytes) < 256:
        return None
    try:
        n_signals = int(header_bytes[252:256].strip())
    except ValueError:
        return None

    needed = 256 + 16 * n_signals
    if len(header_bytes) < needed:
        return None

    labels_raw = header_bytes[256:needed]
    labels = [
        labels_raw[i * 16:(i + 1) * 16].decode("ascii", errors="ignore").strip()
        for i in range(n_signals)
    ]
    eeg_labels = [l for l in labels if l.upper() not in NON_EEG_LABELS]
    return eeg_labels if eeg_labels else None


def parse_eeg_json(json_text):
    """Parse an *_eeg.json sidecar file."""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return {}

    result = {}
    if "SamplingFrequency" in data:
        result["sampling_rate"] = data["SamplingFrequency"]
    if "RecordingDuration" in data:
        try:
            result["duration"] = float(data["RecordingDuration"])
        except (ValueError, TypeError):
            pass
    if "EEGReference" in data:
        ref = str(data["EEGReference"]).strip()
        if ref and ref.lower() not in ("n/a", "na"):
            result["reference"] = ref
    if "Manufacturer" in data:
        mfr = str(data["Manufacturer"]).strip()
        if mfr and mfr.lower() not in ("n/a", "na"):
            result["manufacturer"] = mfr
    if "PowerLineFrequency" in data:
        try:
            result["powerline_freq"] = float(data["PowerLineFrequency"])
        except (ValueError, TypeError):
            pass
    return result


def extract_task_from_filename(fname):
    """Extract task name from a BIDS filename (e.g. 'sub-01_task-rest_eeg.set' -> 'rest')."""
    for part in fname.split("_"):
        if part.startswith("task-"):
            return part[5:]
    return None


def compute_file_stats(blob_list):
    """Compute file statistics from a list of (relative_path, size) tuples.

    Returns dict with ext_counts, total_size, eeg_formats, task breakdowns.
    """
    import os

    ext_counts = Counter()
    total_size = 0
    eeg_formats = set()
    task_sizes = {}
    task_file_counts = {}
    task_subjects = {}

    for rel_path, fsize in blob_list:
        fname = rel_path.rsplit("/", 1)[-1] if "/" in rel_path else rel_path
        ext = os.path.splitext(fname)[1].lower()
        ext_counts[ext] += 1
        total_size += fsize
        if ext in EEG_PRIMARY_FORMATS:
            eeg_formats.add(EEG_FORMAT_EXTENSIONS.get(ext, ext))

        task_name = extract_task_from_filename(fname)
        if task_name:
            task_sizes[task_name] = task_sizes.get(task_name, 0) + fsize
            task_file_counts[task_name] = task_file_counts.get(task_name, 0) + 1
            parts = rel_path.split("/")
            sub = parts[0] if parts and parts[0].startswith("sub-") else ""
            if sub:
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


def aggregate_stats(all_datasets):
    """Aggregate statistics across all datasets."""
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


def format_report(all_datasets, agg):
    """Format the full text report. Returns report as a string."""
    lines = []
    sep = "=" * 80
    subsep = "-" * 80

    lines.append(sep)
    lines.append("  OPENNEURO EEG DATASET STATISTICS")
    lines.append(sep)

    lines.append(f"\n  Datasets:       {agg['n_datasets']}")
    lines.append(f"  Total subjects: {agg['total_subjects']}")
    lines.append(f"  Total files:    {agg['total_files']:,}")
    lines.append(f"  Total size:     {human_size(agg['total_size'])}")
    lines.append(f"  Unique tasks:   {agg['n_tasks']}")

    lines.append(f"\n{subsep}")
    lines.append("  PER-DATASET SUMMARY")
    lines.append(subsep)
    header = f"  {'ID':<12} {'Name':<40} {'Subj':>5} {'Ch':>6} {'SR (Hz)':>10} {'Files':>7} {'Size':>10} {'Format':<25}"
    lines.append(header)
    lines.append(f"  {'-'*12} {'-'*40} {'-'*5} {'-'*6} {'-'*10} {'-'*7} {'-'*10} {'-'*25}")
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
        lines.append(
            f"  {d['id']:<12} {name:<40} {d['n_subjects']:>5} "
            f"{ch_str:>6} {sr_str:>10} {d['n_files']:>7} "
            f"{human_size(d['total_size']):>10} {fmt:<25}"
        )

    lines.append(f"\n{subsep}")
    lines.append("  DEMOGRAPHICS")
    lines.append(subsep)
    if agg["age_stats"]:
        a = agg["age_stats"]
        lines.append(f"  Age (n={a['count']:,} subjects with age data):")
        lines.append(f"    Range:  {a['min']:.1f} - {a['max']:.1f}")
        lines.append(f"    Mean:   {a['mean']:.1f}")
        lines.append(f"    Median: {a['median']:.1f}")
        lines.append(f"    Stdev:  {a['stdev']:.1f}")
    else:
        lines.append("  No age data available.")

    total_sex = sum(agg["sex_counts"].values())
    lines.append(f"\n  Sex/Gender (n={total_sex:,}):")
    for label in ("M", "F", "Other"):
        count = agg["sex_counts"].get(label, 0)
        pct = (count / total_sex * 100) if total_sex else 0
        lines.append(f"    {label:<8} {count:>6}  ({pct:.1f}%)")

    lines.append(f"\n{subsep}")
    lines.append("  RECORDING PARAMETERS")
    lines.append(subsep)

    lines.append("\n  Sampling frequencies (Hz) — datasets using each:")
    for sr, count in sorted(agg["sampling_rates"].items()):
        lines.append(f"    {sr:>8} Hz : {count} dataset(s)")

    lines.append(f"\n  Total channel counts — datasets using each:")
    for cc, count in sorted(agg["channel_counts"].items()):
        lines.append(f"    {cc:>4} ch : {count} dataset(s)")

    if agg["duration_stats"]:
        dur = agg["duration_stats"]
        lines.append(f"\n  Recording duration (seconds):")
        lines.append(f"    Range:  {dur['min']:.1f} - {dur['max']:.1f}")
        lines.append(f"    Mean:   {dur['mean']:.1f}")
        lines.append(f"    Median: {dur['median']:.1f}")

    lines.append("\n  EEG formats — datasets using each:")
    for fmt, count in agg["eeg_formats"].most_common():
        lines.append(f"    {fmt:<25} : {count} dataset(s)")

    lines.append("\n  EEG references — datasets using each:")
    for ref, count in agg["references"].most_common():
        lines.append(f"    {ref:<30} : {count} dataset(s)")

    lines.append("\n  Manufacturers — datasets using each:")
    for mfr, count in agg["manufacturers"].most_common():
        lines.append(f"    {mfr:<35} : {count} dataset(s)")

    lines.append(f"\n{subsep}")
    lines.append("  FILE TYPE BREAKDOWN")
    lines.append(subsep)
    for ext, count in agg["ext_counts"].most_common(20):
        ext_label = ext if ext else "(no ext)"
        lines.append(f"    {ext_label:<15} : {count:>8,}")

    lines.append(f"\n{subsep}")
    lines.append("  EEG CHANNEL LISTS")
    lines.append(subsep)

    montage_groups = {}
    for d in sorted(all_datasets, key=lambda x: x["id"]):
        ch_key = tuple(d["eeg_channels"]) if d["eeg_channels"] else ()
        if ch_key:
            montage_groups.setdefault(ch_key, []).append(d["id"])

    for d in sorted(all_datasets, key=lambda x: x["id"]):
        if not d["eeg_channels"]:
            lines.append(f"\n  {d['id']}  ({d['name'][:50]})")
            lines.append(f"    (no EEG channel info)")
            continue
        ch_key = tuple(d["eeg_channels"])
        shared = montage_groups[ch_key]
        lines.append(f"\n  {d['id']}  ({d['name'][:50]})")
        lines.append(f"    {len(d['eeg_channels'])} EEG channels: {', '.join(d['eeg_channels'])}")
        if len(shared) > 1:
            others = [s for s in shared if s != d["id"]]
            lines.append(f"    ** Same montage as: {', '.join(others)}")

    lines.append(f"\n{subsep}")
    lines.append(f"  TASKS ({agg['n_tasks']} unique)")
    lines.append(subsep)
    for task in agg["tasks"]:
        lines.append(f"    {task}")

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


def format_csv(all_datasets):
    """Format CSV output. Returns CSV content as a string."""
    fieldnames = [
        "id", "name", "task", "license", "bids_version",
        "n_subjects",
        "age_min", "age_max", "age_mean", "age_median",
        "n_male", "n_female", "n_other_sex",
        "sampling_rates", "channel_count", "eeg_channels",
        "eeg_formats", "references", "manufacturers",
        "n_files", "total_size_bytes", "total_size_human",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
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
    return output.getvalue()
