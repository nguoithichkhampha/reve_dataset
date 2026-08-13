#!/usr/bin/env python3
"""Transfer REVE pretraining datasets from S3 to GCS via Storage Transfer Service."""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from google.cloud import storage_transfer_v1

from datasets_config import ALL_SOURCES

OPENNEURO_S3_BUCKET = "openneuro.org"


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_s3_path(s3_path):
    """Parse 's3://bucket/path/' into (bucket, path)."""
    stripped = s3_path.removeprefix("s3://")
    bucket, _, path = stripped.partition("/")
    return bucket, path


def get_s3_info(source, dataset_info):
    """Return (s3_bucket, s3_path) for a dataset."""
    if source == "openneuro":
        return OPENNEURO_S3_BUCKET, f"{dataset_info['id']}/"
    s3_path = dataset_info.get("s3_path")
    if s3_path:
        return parse_s3_path(s3_path)
    return None, None


def create_and_run_transfer(
    client, gcp_project, s3_bucket, s3_path, gcs_bucket, gcs_path,
    aws_access_key_id, aws_secret_access_key,
):
    transfer_job = storage_transfer_v1.TransferJob(
        project_id=gcp_project,
        transfer_spec=storage_transfer_v1.TransferSpec(
            aws_s3_data_source=storage_transfer_v1.AwsS3Data(
                bucket_name=s3_bucket,
                path=s3_path,
                aws_access_key=storage_transfer_v1.AwsAccessKey(
                    access_key_id=aws_access_key_id,
                    secret_access_key=aws_secret_access_key,
                ),
            ),
            gcs_data_sink=storage_transfer_v1.GcsData(
                bucket_name=gcs_bucket,
                path=gcs_path,
            ),
        ),
        status=storage_transfer_v1.TransferJob.Status.ENABLED,
    )

    result = client.create_transfer_job(
        storage_transfer_v1.CreateTransferJobRequest(transfer_job=transfer_job)
    )

    operation = client.run_transfer_job(
        storage_transfer_v1.RunTransferJobRequest(
            job_name=result.name,
            project_id=gcp_project,
        )
    )

    operation.result()
    return result.name


def transfer_dataset(source, dataset_info, gcs_bucket, gcs_prefix, gcp_project,
                     aws_access_key_id, aws_secret_access_key):
    dataset_id = dataset_info["id"]
    s3_bucket, s3_path = get_s3_info(source, dataset_info)

    if not s3_bucket:
        print(f"[{timestamp()}] SKIP   {source}/{dataset_id} — no S3 path configured")
        return dataset_id, -1

    gcs_path = f"{gcs_prefix}{source}/{dataset_id}/"

    print(f"[{timestamp()}] START  {source}/{dataset_id}")
    print(f"  s3://{s3_bucket}/{s3_path} → gs://{gcs_bucket}/{gcs_path}")

    try:
        client = storage_transfer_v1.StorageTransferServiceClient()
        job_name = create_and_run_transfer(
            client, gcp_project, s3_bucket, s3_path, gcs_bucket, gcs_path,
            aws_access_key_id, aws_secret_access_key,
        )
        print(f"[{timestamp()}] DONE   {source}/{dataset_id} (job: {job_name})")
        return dataset_id, 0
    except Exception as exc:
        print(f"[{timestamp()}] FAILED {source}/{dataset_id} — {exc}")
        return dataset_id, 1


def collect_tasks(sources, dataset_filter):
    tasks = []
    for source in sources:
        registry = ALL_SOURCES[source]
        for ds_id, ds_info in registry.items():
            if dataset_filter and ds_id not in dataset_filter:
                continue
            tasks.append((source, ds_info))
    return tasks


def run_dry(tasks, gcs_bucket, gcs_prefix):
    print(f"\nDry run — {len(tasks)} dataset(s) would be transferred to gs://{gcs_bucket}/{gcs_prefix}\n")
    for source, ds_info in tasks:
        s3_bucket, s3_path = get_s3_info(source, ds_info)
        if s3_bucket:
            gcs_path = f"{gcs_prefix}{source}/{ds_info['id']}/"
            print(f"  [{source:>10}]  {ds_info['id']:20s}  s3://{s3_bucket}/{s3_path}")
            print(f"             {'':20s}  → gs://{gcs_bucket}/{gcs_path}")
        else:
            print(f"  [{source:>10}]  {ds_info['id']:20s}  (no S3 path — will skip)")
    print()


def run_transfers(tasks, gcs_bucket, gcs_prefix, gcp_project,
                  aws_access_key_id, aws_secret_access_key, max_workers):
    succeeded, failed, skipped = [], [], []
    t0 = time.time()

    print(f"\n[{timestamp()}] Transferring {len(tasks)} dataset(s) "
          f"to gs://{gcs_bucket}/{gcs_prefix} with {max_workers} worker(s)\n")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for source, ds_info in tasks:
            future = pool.submit(
                transfer_dataset,
                source, ds_info, gcs_bucket, gcs_prefix, gcp_project,
                aws_access_key_id, aws_secret_access_key,
            )
            futures[future] = (source, ds_info)

        for future in as_completed(futures):
            source, ds_info = futures[future]
            try:
                ds_id, returncode = future.result()
                if returncode == 0:
                    succeeded.append(f"{source}/{ds_id}")
                elif returncode == -1:
                    skipped.append(f"{source}/{ds_id}")
                else:
                    failed.append(f"{source}/{ds_id}")
            except Exception as exc:
                failed.append(f"{source}/{ds_info['id']} ({exc})")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Finished in {elapsed:.0f}s")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print("\nFailed datasets:")
        for name in failed:
            print(f"  - {name}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer REVE pretraining datasets from S3 to GCS "
                    "via Google Storage Transfer Service.",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        required=True,
        choices=["tuh", "physionet", "openneuro"],
        help="Data source(s) to transfer. TUH has no S3 bucket and will be skipped.",
    )
    parser.add_argument(
        "--gcs-bucket",
        required=True,
        help="Destination GCS bucket name (without gs:// prefix).",
    )
    parser.add_argument(
        "--gcs-prefix",
        default="",
        help="Path prefix inside the GCS bucket (e.g. 'reve-pretrain/').",
    )
    parser.add_argument(
        "--gcp-project",
        default=os.environ.get("GCP_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT")),
        help="GCP project ID (default: GCP_PROJECT or GOOGLE_CLOUD_PROJECT env).",
    )
    parser.add_argument(
        "--aws-access-key-id",
        default=os.environ.get("AWS_ACCESS_KEY_ID"),
        help="AWS access key ID for S3 read access (default: AWS_ACCESS_KEY_ID env). "
             "Required even for public buckets — STS needs credentials to list objects.",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        default=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        help="AWS secret access key (default: AWS_SECRET_ACCESS_KEY env).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel transfer jobs (default: 4).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Transfer only these dataset IDs (filters across all selected sources).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List datasets and transfer paths without starting transfers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    gcs_bucket = args.gcs_bucket.removeprefix("gs://").rstrip("/")
    gcs_prefix = args.gcs_prefix
    if gcs_prefix and not gcs_prefix.endswith("/"):
        gcs_prefix += "/"

    tasks = collect_tasks(
        sources=args.source,
        dataset_filter=set(args.datasets) if args.datasets else None,
    )

    if not tasks:
        print("No datasets matched the given source/filter combination.")
        sys.exit(1)

    if args.dry_run:
        run_dry(tasks, gcs_bucket, gcs_prefix)
        return

    if not args.gcp_project:
        print("Error: --gcp-project is required (or set GCP_PROJECT env).")
        sys.exit(1)
    if not args.aws_access_key_id or not args.aws_secret_access_key:
        print("Error: --aws-access-key-id and --aws-secret-access-key are required "
              "(or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env).\n"
              "STS needs AWS credentials even for public S3 buckets.")
        sys.exit(1)

    run_transfers(
        tasks, gcs_bucket, gcs_prefix, args.gcp_project,
        args.aws_access_key_id, args.aws_secret_access_key, args.max_workers,
    )


if __name__ == "__main__":
    main()
