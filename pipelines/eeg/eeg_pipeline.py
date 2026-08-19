"""Apache Beam pipeline for EEG preprocessing.

Produces one HDF5 per dataset with all recordings, structured for training
dataloader filtering by n_channels, task, and subject.

Usage:
    # Local (DirectRunner) — test with a single dataset:
    python -m pipelines.eeg.eeg_pipeline \
        --bucket emotiv-reve-data \
        --prefix openneuro/ \
        --output-prefix preprocessed/openneuro/ \
        --direct_num_workers=0 \
        --datasets ds002680

    # Dataflow:
    python -m pipelines.eeg.eeg_pipeline \
        --bucket emotiv-reve-data \
        --prefix openneuro/ \
        --output-prefix preprocessed/openneuro/ \
        --runner DataflowRunner \
        --project emotivml \
        --region us-central1 \
        --temp_location gs://emotiv-reve-data/dataflow-temp/ \
        --setup_file=./setup.py \
        --machine_type n1-highmem-4 \
        --max_num_workers 64 \
        --disk_size_gb 250 \
        --disk_type compute.googleapis.com/projects/emotivml/zones/us-central1-a/diskTypes/pd-ssd \
        --number_of_worker_harness_threads 1 \
        --no-wait --target-sfreq 128 \
        --exclude-datasets ds004395 ds004706 ds004582 ds004356 ds004561 ds005089 ds003004 ds003825 ds003885 ds003887 ds004357 ds004477 ds005273 ds005586 ds005697
"""

import json
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from pipelines.eeg.file_groups import discover_file_groups
from pipelines.eeg.preprocessing import MergeDatasetHDF5Fn, PreprocessEEGFn

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 20 * 1024**3


class UpsertManifestFn(beam.DoFn):
    """Read existing manifest CSV from GCS, replace rows for re-processed
    datasets, append new rows, and write back."""

    def __init__(self, bucket_name, blob_path, header, key_column, project=None):
        self._bucket_name = bucket_name
        self._blob_path = blob_path
        self._header = header
        self._key_column = key_column
        self._project = project

    def setup(self):
        from google.cloud import storage
        self._client = storage.Client(project=self._project)

    def process(self, new_rows):
        if not new_rows:
            return
        bucket = self._client.bucket(self._bucket_name)
        blob = bucket.blob(self._blob_path)

        key_idx = self._header.split(",").index(self._key_column)
        new_keys = {row.split(",")[key_idx] for row in new_rows}

        existing_rows = []
        if blob.exists():
            text = blob.download_as_text()
            for line in text.strip().split("\n"):
                if not line or line == self._header:
                    continue
                key = line.split(",")[key_idx]
                if key not in new_keys:
                    existing_rows.append(line)

        all_rows = existing_rows + new_rows
        content = self._header + "\n" + "\n".join(all_rows) + "\n"
        blob.upload_from_string(content, content_type="text/csv")
        logger.info(
            "Manifest updated: %d existing + %d new = %d rows",
            len(existing_rows), len(new_rows), len(all_rows),
        )


class DiscoverFileGroupsFn(beam.DoFn):
    """Discover all EEG file groups for a dataset."""

    def __init__(self, bucket_name, prefix, project=None):
        self._bucket_name = bucket_name
        self._prefix = prefix
        self._project = project

    def process(self, dataset_id):
        logger.info("Discovering file groups for %s", dataset_id)
        try:
            groups = discover_file_groups(
                self._bucket_name, dataset_id, self._prefix,
                project=self._project,
            )
            logger.info("%s: found %d recording(s)", dataset_id, len(groups))
            for fg in groups:
                yield fg.to_dict()
        except Exception as e:
            logger.error("Discovery failed for %s: %s", dataset_id, e)


def add_eeg_options(parser):
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--prefix", default="openneuro/", help="Prefix within bucket")
    parser.add_argument(
        "--output-prefix", default="preprocessed/openneuro/",
        help="Output prefix within the same bucket",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Specific dataset IDs (default: all from registry)",
    )
    parser.add_argument(
        "--max-file-bytes", type=int, default=MAX_FILE_BYTES,
        help="Skip recordings larger than this (default: 20 GB)",
    )
    parser.add_argument(
        "--manifest-output", default=None,
        help="GCS path for the output manifest JSONL file",
    )
    parser.add_argument(
        "--gcp-project", default="emotivml",
        help="GCP project ID for storage client (default: emotivml)",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Submit the Dataflow job and exit without waiting for completion",
    )
    parser.add_argument(
        "--target-sfreq", type=float, default=None,
        help="Resample all recordings to this frequency (Hz). If not set, keep original.",
    )
    parser.add_argument(
        "--shard-datasets", nargs="*", default=None,
        help="Datasets to shard for parallel merge, format: DATASET_ID:N_SHARDS (e.g. ds004395:8)",
    )
    parser.add_argument(
        "--exclude-datasets", nargs="*", default=None,
        help="Dataset IDs to exclude from processing (e.g. ds004395)",
    )


def run(argv=None):
    import argparse
    import sys
    sys.path.insert(0, ".")
    from datasets_config import OPENNEURO_DATASETS

    parser = argparse.ArgumentParser()
    add_eeg_options(parser)
    known_args, pipeline_args = parser.parse_known_args(argv)

    dataset_ids = known_args.datasets or list(OPENNEURO_DATASETS.keys())
    if known_args.exclude_datasets:
        exclude = set(known_args.exclude_datasets)
        dataset_ids = [d for d in dataset_ids if d not in exclude]
    max_bytes = known_args.max_file_bytes

    shard_map = {}
    if known_args.shard_datasets:
        for spec in known_args.shard_datasets:
            ds_id, n = spec.split(":")
            shard_map[ds_id] = int(n)

    manifest_path = known_args.manifest_output
    if not manifest_path:
        manifest_path = (
            f"gs://{known_args.bucket}/{known_args.output_prefix}manifest"
        )

    use_direct = not any(arg.startswith("--runner") for arg in pipeline_args)

    pipeline_options = PipelineOptions(pipeline_args, pipeline_type_check=False)

    if use_direct:
        from apache_beam.runners.direct.direct_runner import BundleBasedDirectRunner
        p = beam.Pipeline(runner=BundleBasedDirectRunner(), options=pipeline_options)
    else:
        p = beam.Pipeline(options=pipeline_options)

    preprocessed = (
        p
        | "CreateDatasetIDs" >> beam.Create(dataset_ids)
        | "DiscoverFileGroups" >> beam.ParDo(
            DiscoverFileGroupsFn(
                known_args.bucket, known_args.prefix,
                project=known_args.gcp_project,
            )
        )
        | "FilterBySize" >> beam.Filter(
            lambda fg: fg["total_bytes"] <= max_bytes
        )
        | "RedistributeRecordings" >> beam.Reshuffle()
        | "Preprocess" >> beam.ParDo(
            PreprocessEEGFn(
                known_args.bucket,
                known_args.prefix,
                known_args.output_prefix,
                project=known_args.gcp_project,
                target_sfreq=known_args.target_sfreq,
                shard_map=shard_map,
            )
        ).with_outputs("failed", main="success")
    )

    merge_outputs = (
        preprocessed.success
        | "GroupByDataset" >> beam.GroupByKey()
        | "MergeHDF5" >> beam.ParDo(
            MergeDatasetHDF5Fn(
                known_args.bucket,
                known_args.output_prefix,
                project=known_args.gcp_project,
            )
        ).with_outputs("manifest", main="summary")
    )

    # Write per-dataset summary
    (
        merge_outputs.summary
        | "SummaryToJSON" >> beam.Map(json.dumps)
        | "WriteSummary" >> beam.io.WriteToText(
            manifest_path,
            file_name_suffix=".jsonl",
            shard_name_template="",
        )
    )

    MANIFEST_HEADER = "dataset_id,h5_file,h5_path,subject,session,task,run,n_channels,sfreq,n_samples,duration_s"
    manifest_csv_blob = f"{known_args.output_prefix}manifest_recordings.csv"

    def manifest_to_csv_row(row):
        return ",".join(str(row.get(col, "")) for col in MANIFEST_HEADER.split(","))

    (
        merge_outputs.manifest
        | "ManifestToCSV" >> beam.Map(manifest_to_csv_row)
        | "CollectRows" >> beam.combiners.ToList()
        | "UpsertManifest" >> beam.ParDo(
            UpsertManifestFn(
                known_args.bucket,
                manifest_csv_blob,
                MANIFEST_HEADER,
                key_column="dataset_id",
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
