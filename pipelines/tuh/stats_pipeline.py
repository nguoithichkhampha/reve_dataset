"""Apache Beam pipeline for collecting TUH EEG dataset statistics.

Reads EDF headers from GCS (range reads, no full downloads) to collect
channel configs, sampling rates, durations, montage distribution, and sizes.

Usage:
    # Local (DirectRunner):
    python -m pipelines.tuh.stats_pipeline \
        --bucket emotiv-reve-data \
        --prefix tuh/tueg/v2.0.2/ \
        --output-prefix gs://emotiv-reve-data/pipeline-output/tuh-stats/

    # Test with specific groups:
    python -m pipelines.tuh.stats_pipeline \
        --bucket emotiv-reve-data \
        --prefix tuh/tueg/v2.0.2/ \
        --output-prefix gs://emotiv-reve-data/pipeline-output/tuh-stats/ \
        --groups 000 001

    # Dataflow:
    python -m pipelines.tuh.stats_pipeline \
        --bucket emotiv-reve-data \
        --prefix tuh/tueg/v2.0.2/ \
        --output-prefix gs://emotiv-reve-data/pipeline-output/tuh-stats/ \
        --runner DataflowRunner \
        --project emotivml \
        --region us-central1 \
        --temp_location gs://emotiv-reve-data/dataflow-temp/ \
        --setup_file ./setup.py \
        --machine_type n1-standard-2 \
        --max_num_workers 8
"""

import io
import json
import logging
import re
from collections import Counter

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from pipelines.gcs_fs import GCSDatasetFS
from pipelines.tuh.preprocessing import _clean_channel_name, _is_eeg_channel

logger = logging.getLogger(__name__)

ALL_GROUPS = [f"{i:03d}" for i in range(150)]

_PATH_RE = re.compile(
    r"edf/(?P<group>\d{3})/(?P<patient>[^/]+)/"
    r"(?P<session>s\d+)_[^/]+/"
    r"(?P<montage>[^/]+)/"
    r"[^/]+_(?P<token>t\d+)\.edf$"
)


def _parse_edf_header(header_bytes):
    """Parse an EDF header from raw bytes.

    Returns dict with channel_names, sfreq, duration_s, n_channels,
    or None if parsing fails.
    """
    try:
        if len(header_bytes) < 256:
            return None

        n_records = int(header_bytes[236:244].decode("ascii", errors="replace").strip())
        record_duration = float(header_bytes[244:252].decode("ascii", errors="replace").strip())
        n_channels = int(header_bytes[252:256].decode("ascii", errors="replace").strip())

        needed = 256 + n_channels * 256
        if len(header_bytes) < needed:
            return None

        channel_names = []
        for i in range(n_channels):
            start = 256 + i * 16
            name = header_bytes[start:start + 16].decode("ascii", errors="replace").strip()
            channel_names.append(name)

        samples_offset = 256 + n_channels * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
        sfreqs = set()
        for i in range(n_channels):
            start = samples_offset + i * 8
            samples_per_record = int(
                header_bytes[start:start + 8].decode("ascii", errors="replace").strip()
            )
            if record_duration > 0:
                sfreqs.add(samples_per_record / record_duration)

        duration_s = n_records * record_duration if n_records > 0 else 0

        return {
            "channel_names": channel_names,
            "n_channels": n_channels,
            "sfreqs": sorted(sfreqs),
            "sfreq": max(sfreqs) if sfreqs else 0,
            "duration_s": duration_s,
        }
    except (ValueError, UnicodeDecodeError):
        return None


def _human_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


class CollectGroupStatsFn(beam.DoFn):
    """Collect stats for one TUH group (e.g. '000') by reading EDF headers."""

    def __init__(self, bucket_name, prefix, project=None):
        self._bucket_name = bucket_name
        self._prefix = prefix
        self._project = project

    def process(self, group_id):
        from collections import defaultdict

        fs = GCSDatasetFS(self._bucket_name, self._prefix, project=self._project)

        logger.info("Collecting stats for group %s", group_id)

        patients = set()
        sessions = set()
        montage_counts = Counter()
        sfreq_counts = Counter()
        total_duration = 0.0
        total_size = 0
        n_files = 0
        n_parse_errors = 0

        per_nch = defaultdict(lambda: {
            "n_files": 0,
            "patients": set(),
            "total_size": 0,
            "total_duration_s": 0.0,
            "channel_names": set(),
            "sfreqs": Counter(),
            "montages": Counter(),
        })

        for rel_path, size in fs.list_blobs(rel_prefix=f"edf/{group_id}/", suffix=".edf"):
            if size == 0:
                continue

            m = _PATH_RE.search(rel_path)
            if not m:
                continue

            patient = m.group("patient")
            montage = m.group("montage")
            patients.add(patient)
            sessions.add(f"{patient}/{m.group('session')}")
            montage_counts[montage] += 1
            total_size += size
            n_files += 1

            header_size = min(size, 32768)
            try:
                header_bytes = fs.read_bytes(rel_path, start=0, end=header_size - 1)
                parsed = _parse_edf_header(header_bytes)
                if parsed:
                    sfreq_counts[parsed["sfreq"]] += 1
                    total_duration += parsed["duration_s"]

                    eeg_chs = [
                        _clean_channel_name(ch)
                        for ch in parsed["channel_names"]
                        if _is_eeg_channel(ch)
                    ]
                    n_ch = len(eeg_chs)

                    bucket = per_nch[n_ch]
                    bucket["n_files"] += 1
                    bucket["patients"].add(patient)
                    bucket["total_size"] += size
                    bucket["total_duration_s"] += parsed["duration_s"]
                    bucket["channel_names"].update(eeg_chs)
                    bucket["sfreqs"][parsed["sfreq"]] += 1
                    bucket["montages"][montage] += 1
                else:
                    n_parse_errors += 1
            except Exception:
                n_parse_errors += 1

        per_nch_serialized = {}
        for n_ch, bucket in sorted(per_nch.items()):
            per_nch_serialized[n_ch] = {
                "n_files": bucket["n_files"],
                "n_patients": len(bucket["patients"]),
                "total_size": bucket["total_size"],
                "total_duration_h": round(bucket["total_duration_s"] / 3600, 2),
                "eeg_channels": sorted(bucket["channel_names"]),
                "sfreqs": dict(bucket["sfreqs"]),
                "montages": dict(bucket["montages"]),
            }

        result = {
            "group_id": group_id,
            "n_patients": len(patients),
            "n_sessions": len(sessions),
            "n_files": n_files,
            "total_size": total_size,
            "total_duration_s": total_duration,
            "total_duration_h": round(total_duration / 3600, 2),
            "montage_counts": dict(montage_counts),
            "sfreq_counts": dict(sfreq_counts),
            "per_channel_count": per_nch_serialized,
            "n_parse_errors": n_parse_errors,
        }

        logger.info(
            "Group %s: %d patients, %d files, %s, %.1f hours",
            group_id, len(patients), n_files,
            _human_size(total_size), total_duration / 3600,
        )
        yield result


class FormatTUHOutputFn(beam.DoFn):
    """Aggregate group stats and write CSV + report + summary to GCS."""

    def __init__(self, bucket_name, output_prefix, project=None):
        self._bucket_name = bucket_name
        self._output_prefix = output_prefix
        self._project = project

    def process(self, all_groups):
        from collections import defaultdict
        from google.cloud import storage

        all_groups = sorted(all_groups, key=lambda g: g["group_id"])

        total_patients = sum(g["n_patients"] for g in all_groups)
        total_sessions = sum(g["n_sessions"] for g in all_groups)
        total_files = sum(g["n_files"] for g in all_groups)
        total_size = sum(g["total_size"] for g in all_groups)
        total_duration = sum(g["total_duration_s"] for g in all_groups)
        total_errors = sum(g["n_parse_errors"] for g in all_groups)

        montage_totals = Counter()
        sfreq_totals = Counter()
        agg_per_nch = defaultdict(lambda: {
            "n_files": 0,
            "n_patients": 0,
            "total_size": 0,
            "total_duration_h": 0.0,
            "channel_names": set(),
            "sfreqs": Counter(),
            "montages": Counter(),
        })
        for g in all_groups:
            for m, c in g["montage_counts"].items():
                montage_totals[m] += c
            for s, c in g["sfreq_counts"].items():
                sfreq_totals[s] += c
            for n_ch_str, info in g.get("per_channel_count", {}).items():
                n_ch = int(n_ch_str)
                agg = agg_per_nch[n_ch]
                agg["n_files"] += info["n_files"]
                agg["n_patients"] += info["n_patients"]
                agg["total_size"] += info["total_size"]
                agg["total_duration_h"] += info["total_duration_h"]
                agg["channel_names"].update(info.get("eeg_channels", []))
                for s, c in info.get("sfreqs", {}).items():
                    agg["sfreqs"][s] += c
                for m, c in info.get("montages", {}).items():
                    agg["montages"][m] += c

        csv_content = self._format_csv(all_groups)
        report_content = self._format_report(
            all_groups, total_patients, total_sessions, total_files,
            total_size, total_duration, montage_totals, sfreq_totals,
            agg_per_nch, total_errors,
        )

        client = storage.Client(project=self._project)
        bucket = client.bucket(self._bucket_name)

        csv_blob = bucket.blob(f"{self._output_prefix}stats.csv")
        csv_blob.upload_from_string(csv_content, content_type="text/csv")

        report_blob = bucket.blob(f"{self._output_prefix}report.txt")
        report_blob.upload_from_string(report_content, content_type="text/plain")

        per_nch_summary = {}
        for n_ch in sorted(agg_per_nch):
            agg = agg_per_nch[n_ch]
            per_nch_summary[str(n_ch)] = {
                "n_files": agg["n_files"],
                "n_patients": agg["n_patients"],
                "total_size_bytes": agg["total_size"],
                "total_size_human": _human_size(agg["total_size"]),
                "total_duration_hours": agg["total_duration_h"],
                "eeg_channels": sorted(agg["channel_names"]),
            }

        summary_data = {
            "n_groups": len(all_groups),
            "total_patients": total_patients,
            "total_sessions": total_sessions,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_human": _human_size(total_size),
            "total_duration_hours": round(total_duration / 3600, 2),
            "montage_distribution": dict(montage_totals),
            "sfreq_distribution": {str(k): v for k, v in sfreq_totals.most_common()},
            "per_channel_count": per_nch_summary,
        }
        summary_blob = bucket.blob(f"{self._output_prefix}summary.json")
        summary_blob.upload_from_string(
            json.dumps(summary_data, indent=2), content_type="application/json"
        )

        logger.info(
            "Output written: gs://%s/%sstats.csv, report.txt, summary.json",
            self._bucket_name, self._output_prefix,
        )
        yield summary_data

    def _format_csv(self, all_groups):
        buf = io.StringIO()
        header = [
            "group_id", "channel_count", "n_patients", "n_files",
            "total_size_bytes", "total_size", "duration_hours",
            "montages", "sfreqs", "eeg_channels",
        ]
        buf.write(",".join(header) + "\n")
        for g in all_groups:
            for n_ch_str, info in sorted(g.get("per_channel_count", {}).items(),
                                         key=lambda x: int(x[0])):
                montages = "; ".join(
                    f"{m}({c})" for m, c in sorted(info["montages"].items())
                )
                sfreqs = "; ".join(
                    f"{s}({c})" for s, c in sorted(info["sfreqs"].items())
                )
                eeg_channels = "; ".join(info.get("eeg_channels", []))
                row = [
                    g["group_id"],
                    str(n_ch_str),
                    str(info["n_patients"]),
                    str(info["n_files"]),
                    str(info["total_size"]),
                    _human_size(info["total_size"]),
                    str(info["total_duration_h"]),
                    montages,
                    sfreqs,
                    eeg_channels,
                ]
                buf.write(",".join(row) + "\n")
        return buf.getvalue()

    def _format_report(
        self, all_groups, total_patients, total_sessions, total_files,
        total_size, total_duration, montage_totals, sfreq_totals,
        agg_per_nch, total_errors,
    ):
        lines = [
            "TUH EEG Corpus — Statistics Report",
            "=" * 50,
            "",
            f"Groups processed:      {len(all_groups)}",
            f"Total patients:        {total_patients:,}",
            f"Total sessions:        {total_sessions:,}",
            f"Total EDF files:       {total_files:,}",
            f"Total size:            {_human_size(total_size)}",
            f"Total duration:        {total_duration / 3600:,.1f} hours",
            f"Avg duration/file:     {total_duration / max(total_files, 1) / 60:.1f} min",
            f"Avg sessions/patient:  {total_sessions / max(total_patients, 1):.2f}",
            f"Parse errors:          {total_errors}",
            "",
            "Montage Distribution",
            "-" * 30,
        ]
        for m, c in montage_totals.most_common():
            pct = 100 * c / max(total_files, 1)
            lines.append(f"  {m:20s}  {c:6,} files ({pct:.1f}%)")

        lines += [
            "",
            "Sampling Frequency Distribution",
            "-" * 30,
        ]
        for s, c in sfreq_totals.most_common():
            pct = 100 * c / max(total_files, 1)
            lines.append(f"  {s:>8.1f} Hz  {c:6,} files ({pct:.1f}%)")

        lines += [
            "",
            "EEG Channel Count Distribution",
            "-" * 50,
        ]
        for n_ch in sorted(agg_per_nch):
            agg = agg_per_nch[n_ch]
            pct = 100 * agg["n_files"] / max(total_files, 1)
            ch_list = sorted(agg["channel_names"])
            ch_str = ", ".join(ch_list)
            lines.append(
                f"  {n_ch:2d} ch  {agg['n_files']:6,} files ({pct:5.1f}%)  "
                f"{agg['n_patients']:5,} patients  "
                f"{_human_size(agg['total_size']):>8s}  "
                f"{agg['total_duration_h']:7.1f} h"
            )
            lines.append(f"         [{ch_str}]")

        lines += [
            "",
            "Per-Group Summary",
            "-" * 30,
        ]
        for g in all_groups:
            lines.append(
                f"  {g['group_id']}: {g['n_patients']:3d} patients, "
                f"{g['n_files']:4d} files, "
                f"{_human_size(g['total_size']):>10s}, "
                f"{g['total_duration_h']:7.1f} h"
            )

        lines.append("")
        return "\n".join(lines)


def add_stats_options(parser):
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--prefix", default="tuh/tueg/v2.0.2/",
        help="Prefix within bucket for TUH dataset",
    )
    parser.add_argument(
        "--output-prefix", required=True,
        help="GCS output prefix (e.g. gs://bucket/pipeline-output/tuh-stats/)",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Specific group IDs to process (default: all 000-149)",
    )
    parser.add_argument(
        "--gcp-project", default="emotivml",
        help="GCP project ID for storage client (default: emotivml)",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Submit the Dataflow job and exit without waiting for completion",
    )


def run(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    add_stats_options(parser)
    known_args, pipeline_args = parser.parse_known_args(argv)

    output_prefix = known_args.output_prefix
    if output_prefix.startswith("gs://"):
        parts = output_prefix[5:].split("/", 1)
        output_bucket = parts[0]
        output_path = parts[1] if len(parts) > 1 else ""
    else:
        output_bucket = known_args.bucket
        output_path = output_prefix

    if not output_path.endswith("/"):
        output_path += "/"

    group_ids = known_args.groups or ALL_GROUPS

    use_direct = not any(arg.startswith("--runner") for arg in pipeline_args)

    pipeline_options = PipelineOptions(pipeline_args)

    if use_direct:
        from apache_beam.runners.direct.direct_runner import BundleBasedDirectRunner
        p = beam.Pipeline(runner=BundleBasedDirectRunner(), options=pipeline_options)
    else:
        p = beam.Pipeline(options=pipeline_options)

    stats = (
        p
        | "CreateGroupIDs" >> beam.Create(group_ids)
        | "CollectStats" >> beam.ParDo(
            CollectGroupStatsFn(
                known_args.bucket, known_args.prefix,
                project=known_args.gcp_project,
            )
        )
    )

    (
        stats
        | "GatherAll" >> beam.combiners.ToList()
        | "FormatAndWrite" >> beam.ParDo(
            FormatTUHOutputFn(
                output_bucket, output_path,
                project=known_args.gcp_project,
            )
        )
    )

    result = p.run()
    if known_args.no_wait:
        logger.info("Job submitted. View in Dataflow console.")
    else:
        result.wait_until_finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
