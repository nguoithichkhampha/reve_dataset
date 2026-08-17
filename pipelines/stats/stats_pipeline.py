"""Apache Beam pipeline for collecting OpenNeuro EEG dataset statistics.

Usage:
    # Local (DirectRunner):
    python -m pipelines.stats.stats_pipeline \
        --bucket emotiv-reve-data \
        --prefix openneuro/ \
        --output-prefix gs://emotiv-reve-data/pipeline-output/stats/

    # Dataflow:
    python -m pipelines.stats.stats_pipeline \
        --bucket emotiv-reve-data \
        --prefix openneuro/ \
        --output-prefix gs://emotiv-reve-data/pipeline-output/stats/ \
        --runner DataflowRunner \
        --project emotivml \
        --region us-central1 \
        --temp_location gs://emotiv-reve-data/dataflow-temp/ \
        --setup_file ./setup.py \
        --machine_type n1-standard-2 \
        --max_num_workers 8
"""

import json
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from pipelines.gcs_fs import GCSDatasetFS
from pipelines.stats.parsers import (
    aggregate_stats,
    compute_file_stats,
    format_csv,
    format_report,
    human_size,
    parse_channels_tsv,
    parse_dataset_description,
    parse_edf_header_channels,
    parse_eeg_json,
    parse_participants,
    parse_vhdr_channels,
    select_best_channel_list,
)

logger = logging.getLogger(__name__)


class CollectDatasetStatsFn(beam.DoFn):
    """Collect stats for a single OpenNeuro dataset from GCS."""

    def __init__(self, bucket_name, prefix, project=None):
        self._bucket_name = bucket_name
        self._prefix = prefix
        self._project = project

    def process(self, dataset_id):
        ds_prefix = f"{self._prefix}{dataset_id}"
        fs = GCSDatasetFS(self._bucket_name, ds_prefix, project=self._project)

        logger.info("Processing %s", dataset_id)

        desc = self._parse_description(fs)
        participants = self._parse_participants(fs)
        eeg_params = self._parse_eeg_params(fs)
        eeg_channels = self._get_eeg_channels(fs)
        blob_list = list(fs.list_all_blobs())
        file_stats = compute_file_stats(blob_list)

        subjects = set()
        for rel_path, _ in blob_list:
            parts = rel_path.split("/")
            if parts and parts[0].startswith("sub-"):
                subjects.add(parts[0])
        n_subjects = max(len(subjects), participants["count"])

        result = {
            "id": dataset_id,
            "name": desc.get("name", ""),
            "license": desc.get("license", ""),
            "bids_version": desc.get("bids_version", ""),
            "n_subjects": n_subjects,
            "ages": participants["ages"],
            "sex_counts": dict(participants["sex_counts"]),
            "sampling_rates": eeg_params["sampling_rates"],
            "channel_count": len(eeg_channels) if eeg_channels else None,
            "eeg_channels": eeg_channels or [],
            "durations": eeg_params["durations"],
            "task_durations": eeg_params["task_durations"],
            "tasks": sorted(file_stats["task_sizes"].keys()),
            "references": eeg_params["references"],
            "manufacturers": eeg_params["manufacturers"],
            "eeg_formats": file_stats["eeg_formats"],
            "ext_counts": dict(file_stats["ext_counts"]),
            "total_size": file_stats["total_size"],
            "n_files": sum(file_stats["ext_counts"].values()),
            "task_sizes": file_stats["task_sizes"],
            "task_file_counts": file_stats["task_file_counts"],
            "task_subjects": file_stats["task_subjects"],
        }
        total_dur_h = round(sum(result["durations"]) / 3600, 2) if result["durations"] else 0
        logger.info(
            "%s: %d subjects, %d files, %s, %.1f hours",
            dataset_id, n_subjects, result["n_files"],
            human_size(result["total_size"]), total_dur_h,
        )
        yield result

    def _parse_description(self, fs):
        try:
            text = fs.read_text("dataset_description.json")
            return parse_dataset_description(text)
        except Exception:
            return {}

    def _parse_participants(self, fs):
        try:
            text = fs.read_text("participants.tsv")
            return parse_participants(text)
        except Exception:
            return {"count": 0, "ages": [], "sex_counts": {}}

    def _parse_eeg_params(self, fs):
        from pipelines.stats.parsers import extract_task_from_filename

        sampling_rates = set()
        durations = []
        task_durations = {}
        references = set()
        manufacturers = set()

        for rel_path, _ in fs.list_blobs(suffix="_eeg.json"):
            try:
                text = fs.read_text(rel_path)
                params = parse_eeg_json(text)
                if "sampling_rate" in params:
                    sampling_rates.add(params["sampling_rate"])
                if "duration" in params:
                    dur = params["duration"]
                    durations.append(dur)
                    fname = rel_path.rsplit("/", 1)[-1] if "/" in rel_path else rel_path
                    task = extract_task_from_filename(fname)
                    if task:
                        task_durations[task] = task_durations.get(task, 0.0) + dur
                if "reference" in params:
                    references.add(params["reference"])
                if "manufacturer" in params:
                    manufacturers.add(params["manufacturer"])
            except Exception:
                continue

        return {
            "sampling_rates": sorted(sampling_rates),
            "durations": durations,
            "task_durations": task_durations,
            "references": sorted(references),
            "manufacturers": sorted(manufacturers),
        }

    def _get_eeg_channels(self, fs):
        channel_lists = []
        for rel_path, _ in fs.list_blobs(suffix="_channels.tsv"):
            try:
                text = fs.read_text(rel_path)
                eeg, misc, all_ch = parse_channels_tsv(text)
                channel_lists.append((eeg, misc, all_ch))
            except Exception:
                continue

        if channel_lists:
            result = select_best_channel_list(channel_lists)
            if result:
                return result

        for rel_path, _ in fs.list_blobs(suffix=".vhdr"):
            try:
                text = fs.read_text(rel_path)
                names = parse_vhdr_channels(text)
                if names:
                    return names
            except Exception:
                continue

        for rel_path, size in fs.list_blobs(suffix=".edf"):
            try:
                needed = min(256 + 16 * 512, size)
                header = fs.read_bytes(rel_path, start=0, end=needed - 1)
                names = parse_edf_header_channels(header)
                if names:
                    return names
            except Exception:
                continue

        return []


class FormatOutputFn(beam.DoFn):
    """Format aggregated stats into CSV and report, write to GCS."""

    def __init__(self, bucket_name, output_prefix, project=None):
        self._bucket_name = bucket_name
        self._output_prefix = output_prefix
        self._project = project

    def process(self, all_datasets):
        from collections import Counter
        from google.cloud import storage

        for d in all_datasets:
            d["sex_counts"] = Counter(d.get("sex_counts", {}))
            d["ext_counts"] = Counter(d.get("ext_counts", {}))

        agg = aggregate_stats(all_datasets)

        csv_content = format_csv(all_datasets)
        report_content = format_report(all_datasets, agg)

        client = storage.Client(project=self._project)
        bucket = client.bucket(self._bucket_name)

        csv_blob = bucket.blob(f"{self._output_prefix}stats.csv")
        csv_blob.upload_from_string(csv_content, content_type="text/csv")

        report_blob = bucket.blob(f"{self._output_prefix}report.txt")
        report_blob.upload_from_string(report_content, content_type="text/plain")

        summary_data = {
            "n_datasets": agg["n_datasets"],
            "total_subjects": agg["total_subjects"],
            "total_files": agg["total_files"],
            "total_size_bytes": agg["total_size"],
            "total_size_human": human_size(agg["total_size"]),
            "total_duration_hours": agg["duration_stats"].get("total_hours", 0),
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


def add_stats_options(parser):
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--prefix", default="openneuro/", help="Prefix within bucket")
    parser.add_argument(
        "--output-prefix", required=True,
        help="GCS output prefix (e.g. gs://bucket/pipeline-output/stats/)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Specific dataset IDs to process (default: all from registry)",
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
    import sys
    sys.path.insert(0, ".")
    from datasets_config import OPENNEURO_DATASETS

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

    dataset_ids = known_args.datasets or list(OPENNEURO_DATASETS.keys())

    use_direct = not any(arg.startswith("--runner") for arg in pipeline_args)

    pipeline_options = PipelineOptions(pipeline_args)

    if use_direct:
        from apache_beam.runners.direct.direct_runner import BundleBasedDirectRunner
        p = beam.Pipeline(runner=BundleBasedDirectRunner(), options=pipeline_options)
    else:
        p = beam.Pipeline(options=pipeline_options)

    stats = (
        p
        | "CreateDatasetIDs" >> beam.Create(dataset_ids)
        | "CollectStats" >> beam.ParDo(
            CollectDatasetStatsFn(known_args.bucket, known_args.prefix, project=known_args.gcp_project)
        )
    )

    (
        stats
        | "GatherAll" >> beam.combiners.ToList()
        | "FormatAndWrite" >> beam.ParDo(
            FormatOutputFn(output_bucket, output_path, project=known_args.gcp_project)
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
