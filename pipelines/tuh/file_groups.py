"""Discover EEG recordings from the TUH EEG Corpus in GCS.

TUH path structure:
  edf/{group}/{patient_id}/{session}_{date}/{montage}/{patient_id}_{session}_{token}.edf

Montage types:
  01_tcp_ar   — TCP average reference (channels suffixed -REF)
  02_tcp_le   — TCP linked ears (channels suffixed -LE)
  03_tcp_ar_a — TCP average reference alternate (channels suffixed -REF)
"""

import re
from dataclasses import dataclass

from pipelines.gcs_fs import GCSDatasetFS


@dataclass
class TUHFileGroup:
    group: str
    patient_id: str
    session: str
    montage: str
    token: str
    blob_path: str
    size_bytes: int

    def to_dict(self):
        return {
            "group": self.group,
            "patient_id": self.patient_id,
            "session": self.session,
            "montage": self.montage,
            "token": self.token,
            "blob_path": self.blob_path,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


_PATH_RE = re.compile(
    r"edf/(?P<group>\d{3})/(?P<patient>[^/]+)/"
    r"(?P<session>s\d+)_[^/]+/"
    r"(?P<montage>[^/]+)/"
    r"[^/]+_(?P<token>t\d+)\.edf$"
)


def discover_file_groups(bucket_name, group_id, prefix="tuh/tueg/v2.0.2/", project=None):
    """Discover all EDF recordings for a TUH group (e.g. '000').

    Returns a list of TUHFileGroup objects.
    """
    group_prefix = f"{prefix}edf/{group_id}/"
    fs = GCSDatasetFS(bucket_name, prefix, project=project)

    groups = []
    for rel_path, size in fs.list_blobs(rel_prefix=f"edf/{group_id}/", suffix=".edf"):
        if size == 0:
            continue
        full_path = prefix + rel_path
        m = _PATH_RE.search(full_path)
        if not m:
            continue

        groups.append(TUHFileGroup(
            group=m.group("group"),
            patient_id=m.group("patient"),
            session=m.group("session"),
            montage=m.group("montage"),
            token=m.group("token"),
            blob_path=rel_path,
            size_bytes=size,
        ))

    return groups
