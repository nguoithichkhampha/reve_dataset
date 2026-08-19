"""Discover multi-file EEG recording groups from GCS blob listings."""

import os
from collections import defaultdict
from dataclasses import dataclass, field

from pipelines.gcs_fs import GCSDatasetFS


FORMAT_FILE_GROUPS = {
    "brainvision": {".vhdr", ".eeg", ".vmrk"},
    "eeglab": {".set", ".fdt"},
    "biosemi": {".bdf"},
    "edf": {".edf"},
    "mne": {".fif"},
}

PRIMARY_EXTENSIONS = {
    ".vhdr": "brainvision",
    ".set": "eeglab",
    ".bdf": "biosemi",
    ".edf": "edf",
    ".fif": "mne",
}

AUX_EXTENSIONS = {".eeg", ".vmrk", ".fdt"}


@dataclass
class EEGFileGroup:
    dataset_id: str
    subject: str
    session: str
    task: str
    run: str
    format: str
    primary_blob: str
    aux_blobs: list = field(default_factory=list)
    sidecar_blobs: list = field(default_factory=list)
    total_bytes: int = 0

    def to_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "subject": self.subject,
            "session": self.session,
            "task": self.task,
            "run": self.run,
            "format": self.format,
            "primary_blob": self.primary_blob,
            "aux_blobs": self.aux_blobs,
            "sidecar_blobs": self.sidecar_blobs,
            "total_bytes": self.total_bytes,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _parse_bids_entities(rel_path):
    """Extract BIDS entities from a relative path."""
    parts = rel_path.split("/")
    fname = parts[-1]

    subject = ""
    session = ""
    for p in parts[:-1]:
        if p.startswith("sub-"):
            subject = p
        elif p.startswith("ses-"):
            session = p

    task = ""
    run = ""
    stem = os.path.splitext(fname)[0]
    if fname.endswith(".vhdr") or fname.endswith(".vmrk") or fname.endswith(".eeg"):
        stem = os.path.splitext(fname)[0]

    for entity in stem.split("_"):
        if entity.startswith("task-"):
            task = entity[5:]
        elif entity.startswith("run-"):
            run = entity[4:]

    return subject, session, task, run


def _stem_key(rel_path):
    """Return stem without extension for grouping related files."""
    fname = rel_path.rsplit("/", 1)[-1] if "/" in rel_path else rel_path
    base, _ = os.path.splitext(fname)
    dir_part = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
    return f"{dir_part}/{base}" if dir_part else base


def discover_file_groups(bucket_name, dataset_id, prefix="openneuro/", project=None):
    """Discover all EEG recording file groups for a dataset.

    Returns a list of EEGFileGroup objects.
    """
    ds_prefix = f"{prefix}{dataset_id}"
    fs = GCSDatasetFS(bucket_name, ds_prefix, project=project)

    stems = defaultdict(lambda: {"files": [], "total_size": 0})
    for rel_path, size in fs.list_all_blobs():
        if rel_path.startswith("derivatives/") or "/derivatives/" in rel_path:
            continue
        if "/eeg/" not in f"/{rel_path}/" and not rel_path.endswith(
            tuple(PRIMARY_EXTENSIONS) + tuple(AUX_EXTENSIONS)
        ):
            continue
        key = _stem_key(rel_path)
        stems[key]["files"].append((rel_path, size))
        stems[key]["total_size"] += size

    groups = []
    for stem_key, info in stems.items():
        primary = None
        aux = []
        sidecars = []
        fmt = None

        for rel_path, size in info["files"]:
            ext = os.path.splitext(rel_path)[1].lower()
            if ext in PRIMARY_EXTENSIONS:
                primary = rel_path
                fmt = PRIMARY_EXTENSIONS[ext]
            elif ext in AUX_EXTENSIONS:
                aux.append(rel_path)
            elif ext == ".json":
                sidecars.append(rel_path)

        if not primary:
            continue

        subject, session, task, run = _parse_bids_entities(primary)

        groups.append(EEGFileGroup(
            dataset_id=dataset_id,
            subject=subject,
            session=session,
            task=task,
            run=run,
            format=fmt,
            primary_blob=primary,
            aux_blobs=aux,
            sidecar_blobs=sidecars,
            total_bytes=info["total_size"],
        ))

    return groups
