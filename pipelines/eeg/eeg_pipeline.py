"""Apache Beam pipeline for EEG preprocessing.

Produces one HDF5 per dataset with all recordings, structured for training
dataloader filtering by n_channels, task, and subject.

Usage:
    # Local (DirectRunner) — test with a single dataset:
    python -m pipelines.eeg.eeg_pipeline \
        --bucket emotiv-reve-data \
        --prefix openneuro/ \
        --output-prefix preprocessed/openneuro/ \
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
        --setup_file ./setup.py \
        --machine_type n1-highmem-4 \
        --max_num_workers 32 \
        --disk_size_gb 500 \
        --disk_type pd-ssd \
        --no-wait
"""

import json
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from pipelines.eeg.file_groups import discover_file_groups
from pipelines.eeg.preprocessing import MergeDatasetHDF5Fn, PreprocessEEGFn

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 20 * 1024**3


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


def run(argv=None):
    import argparse
    import sys
    sys.path.insert(0, ".")
    from datasets_config import OPENNEURO_DATASETS

    parser = argparse.ArgumentParser()
    add_eeg_options(parser)
    known_args, pipeline_args = parser.parse_known_args(argv)

    dataset_ids = known_args.datasets or list(OPENNEURO_DATASETS.keys())
    max_bytes = known_args.max_file_bytes

    manifest_path = known_args.manifest_output
    if not manifest_path:
        manifest_path = (
            f"gs://{known_args.bucket}/{known_args.output_prefix}manifest"
        )

    if not any(arg.startswith("--runner") for arg in pipeline_args):
        pipeline_args.append("--runner=DirectRunner")

    pipeline_options = PipelineOptions(pipeline_args)

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
        | "Preprocess" >> beam.ParDo(
            PreprocessEEGFn(
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

    # Write per-recording manifest CSV for dataloader filtering
    MANIFEST_HEADER = "dataset_id,h5_file,h5_path,subject,session,task,run,n_channels,sfreq,n_samples,duration_s"

    def manifest_to_csv_row(row):
        return ",".join(str(row.get(col, "")) for col in MANIFEST_HEADER.split(","))

    (
        merge_outputs.manifest
        | "ManifestToCSV" >> beam.Map(manifest_to_csv_row)
        | "WriteManifestCSV" >> beam.io.WriteToText(
            manifest_path + "_recordings",
            file_name_suffix=".csv",
            shard_name_template="",
            header=MANIFEST_HEADER,
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
