"""Apache Beam pipeline for TUH EEG preprocessing.

Produces one HDF5 per TUH group (000-149) with all recordings, structured
for training dataloader filtering by n_channels, patient, and session.

Usage:
    # Local (DirectRunner) — test with one group:
    python -m pipelines.tuh.tuh_pipeline \
        --bucket emotiv-reve-data \
        --prefix tuh/tueg/v2.0.2/ \
        --output-prefix preprocessed/tuh/ \
        --groups 000

    # Dataflow:
    python -m pipelines.tuh.tuh_pipeline \
        --bucket emotiv-reve-data \
        --prefix tuh/tueg/v2.0.2/ \
        --output-prefix preprocessed/tuh/ \
        --runner DataflowRunner \
        --project emotivml \
        --region us-central1 \
        --temp_location gs://emotiv-reve-data/dataflow-temp/ \
        --setup_file=./setup.py \
        --machine_type n1-highmem-4 \
        --max_num_workers 64 \
        --disk_size_gb 100 \
        --disk_type compute.googleapis.com/projects/emotivml/zones/us-central1-a/diskTypes/pd-ssd \
        --number_of_worker_harness_threads 1 \
        --no-wait --target-sfreq 128
"""

import json
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from pipelines.tuh.file_groups import discover_file_groups
from pipelines.tuh.preprocessing import TUHMergeGroupHDF5Fn, TUHPreprocessEEGFn

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 20 * 1024**3
ALL_GROUPS = [f"{i:03d}" for i in range(150)]


class UpsertManifestFn(beam.DoFn):
    """Read existing manifest CSV from GCS, replace rows for re-processed
    groups, append new rows, and write back."""

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


class DiscoverTUHFileGroupsFn(beam.DoFn):
    """Discover all EDF recordings for a TUH group."""

    def __init__(self, bucket_name, prefix, project=None):
        self._bucket_name = bucket_name
        self._prefix = prefix
        self._project = project

    def process(self, group_id):
        logger.info("Discovering EDF files for group %s", group_id)
        try:
            groups = discover_file_groups(
                self._bucket_name, group_id, self._prefix,
                project=self._project,
            )
            logger.info("Group %s: found %d recording(s)", group_id, len(groups))
            for fg in groups:
                yield fg.to_dict()
        except Exception as e:
            logger.error("Discovery failed for group %s: %s", group_id, e)


def add_tuh_options(parser):
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument(
        "--prefix", default="tuh/tueg/v2.0.2/",
        help="Prefix within bucket for TUH dataset",
    )
    parser.add_argument(
        "--output-prefix", default="preprocessed/tuh/",
        help="Output prefix within the same bucket",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Specific group IDs to process (default: all 000-149)",
    )
    parser.add_argument(
        "--max-file-bytes", type=int, default=MAX_FILE_BYTES,
        help="Skip recordings larger than this (default: 20 GB)",
    )
    parser.add_argument(
        "--manifest-output", default=None,
        help="GCS path for the output manifest file",
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


def run(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    add_tuh_options(parser)
    known_args, pipeline_args = parser.parse_known_args(argv)

    group_ids = known_args.groups or ALL_GROUPS
    max_bytes = known_args.max_file_bytes

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
        | "CreateGroupIDs" >> beam.Create(group_ids)
        | "DiscoverFileGroups" >> beam.ParDo(
            DiscoverTUHFileGroupsFn(
                known_args.bucket, known_args.prefix,
                project=known_args.gcp_project,
            )
        )
        | "FilterBySize" >> beam.Filter(
            lambda fg: fg["size_bytes"] <= max_bytes
        )
        | "RedistributeRecordings" >> beam.Reshuffle()
        | "Preprocess" >> beam.ParDo(
            TUHPreprocessEEGFn(
                known_args.bucket,
                known_args.prefix,
                known_args.output_prefix,
                project=known_args.gcp_project,
                target_sfreq=known_args.target_sfreq,
            )
        ).with_outputs("failed", main="success")
    )

    merge_outputs = (
        preprocessed.success
        | "GroupByGroup" >> beam.GroupByKey()
        | "MergeHDF5" >> beam.ParDo(
            TUHMergeGroupHDF5Fn(
                known_args.bucket,
                known_args.output_prefix,
                project=known_args.gcp_project,
            )
        ).with_outputs("manifest", main="summary")
    )

    (
        merge_outputs.summary
        | "SummaryToJSON" >> beam.Map(json.dumps)
        | "WriteSummary" >> beam.io.WriteToText(
            manifest_path,
            file_name_suffix=".jsonl",
            shard_name_template="",
        )
    )

    MANIFEST_HEADER = "group_id,h5_file,h5_path,patient_id,session,montage,token,n_channels,sfreq,n_samples,duration_s"
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
                key_column="group_id",
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
