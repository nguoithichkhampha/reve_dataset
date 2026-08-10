#!/usr/bin/env python3
"""Download REVE pretraining datasets from TUH, PhysioNet, and OpenNeuro."""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from datasets_config import ALL_SOURCES

TUH_RSYNC_HOST = "www.isip.piconepress.com"
OPENNEURO_S3_BUCKET = "s3://openneuro.org"


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def download_tuh(dataset_info, output_dir, tuh_user):
    dataset_id = dataset_info["id"]
    dest = os.path.join(output_dir, "tuh")
    os.makedirs(dest, exist_ok=True)

    rsync_src = f"{tuh_user}@{TUH_RSYNC_HOST}:~/{dataset_info['rsync_path']}"
    cmd = ["rsync", "-auxvz", rsync_src, dest]

    print(f"[{timestamp()}] START  tuh/{dataset_id}")
    print(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[{timestamp()}] DONE   tuh/{dataset_id}")
    else:
        print(f"[{timestamp()}] FAILED tuh/{dataset_id} (exit {result.returncode})")
    return dataset_id, result.returncode


def download_physionet(dataset_info, output_dir):
    dataset_id = dataset_info["id"]
    dest = os.path.join(output_dir, "physionet", dataset_id)
    os.makedirs(dest, exist_ok=True)

    url = dataset_info["url"]
    cmd = [
        "wget", "-r", "-N", "-c", "-np", "-nH",
        "--cut-dirs=2",
        "-P", dest,
        url,
    ]

    print(f"[{timestamp()}] START  physionet/{dataset_id}")
    print(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[{timestamp()}] DONE   physionet/{dataset_id}")
    else:
        print(f"[{timestamp()}] FAILED physionet/{dataset_id} (exit {result.returncode})")
    return dataset_id, result.returncode


def download_openneuro(dataset_info, output_dir):
    dataset_id = dataset_info["id"]
    dest = os.path.join(output_dir, "openneuro", dataset_id)
    os.makedirs(dest, exist_ok=True)

    s3_src = f"{OPENNEURO_S3_BUCKET}/{dataset_id}/"

    if shutil.which("aws"):
        cmd = ["aws", "s3", "sync", "--no-sign-request", s3_src, dest]
    else:
        cmd = [
            sys.executable, "-m", "openneuro",
            "download", f"--dataset={dataset_id}", f"--target-dir={dest}",
        ]

    print(f"[{timestamp()}] START  openneuro/{dataset_id}")
    print(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[{timestamp()}] DONE   openneuro/{dataset_id}")
    else:
        print(f"[{timestamp()}] FAILED openneuro/{dataset_id} (exit {result.returncode})")
    return dataset_id, result.returncode


def _download_worker(args):
    source, dataset_info, output_dir, tuh_user = args
    if source == "tuh":
        return download_tuh(dataset_info, output_dir, tuh_user)
    elif source == "physionet":
        return download_physionet(dataset_info, output_dir)
    elif source == "openneuro":
        return download_openneuro(dataset_info, output_dir)


def collect_tasks(sources, dataset_filter, tuh_user, output_dir):
    tasks = []
    for source in sources:
        registry = ALL_SOURCES[source]
        for ds_id, ds_info in registry.items():
            if dataset_filter and ds_id not in dataset_filter:
                continue
            tasks.append((source, ds_info, output_dir, tuh_user))
    return tasks


def run_dry(tasks):
    print(f"\nDry run — {len(tasks)} dataset(s) would be downloaded:\n")
    for source, ds_info, _, _ in tasks:
        print(f"  [{source:>10}]  {ds_info['id']:20s}  {ds_info['description']}")
    print()


def run_downloads(tasks, max_workers):
    succeeded, failed = [], []
    t0 = time.time()

    print(f"\n[{timestamp()}] Downloading {len(tasks)} dataset(s) with {max_workers} worker(s)\n")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_worker, t): t for t in tasks}
        for future in as_completed(futures):
            task_info = futures[future]
            source, ds_info = task_info[0], task_info[1]
            try:
                ds_id, returncode = future.result()
                if returncode == 0:
                    succeeded.append(f"{source}/{ds_id}")
                else:
                    failed.append(f"{source}/{ds_id}")
            except Exception as exc:
                failed.append(f"{source}/{ds_info['id']} ({exc})")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Finished in {elapsed:.0f}s")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print("\nFailed datasets:")
        for name in failed:
            print(f"  - {name}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download REVE pretraining datasets from TUH, PhysioNet, and OpenNeuro.",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        required=True,
        choices=["tuh", "physionet", "openneuro"],
        help="Data source(s) to download from.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Base output directory (default: ./data).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel download processes (default: 4).",
    )
    parser.add_argument(
        "--tuh-user",
        default=os.environ.get("TUH_USER", "nedc_tuh_eeg"),
        help="TUH rsync username (default: TUH_USER env or nedc_tuh_eeg).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Download only these dataset IDs (filters across all selected sources).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List datasets without downloading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tasks = collect_tasks(
        sources=args.source,
        dataset_filter=set(args.datasets) if args.datasets else None,  # type: ignore[arg-type]
        tuh_user=args.tuh_user,
        output_dir=args.output_dir,
    )

    if not tasks:
        print("No datasets matched the given source/filter combination.")
        sys.exit(1)

    if args.dry_run:
        run_dry(tasks)
    else:
        run_downloads(tasks, args.max_workers)


if __name__ == "__main__":
    main()
