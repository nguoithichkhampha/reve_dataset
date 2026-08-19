"""GCS filesystem abstraction for BIDS dataset access."""

import logging

from google.cloud import storage

logger = logging.getLogger(__name__)


class GCSDatasetFS:
    """Filesystem-like interface for a BIDS dataset stored in GCS.

    Wraps a bucket + prefix (e.g. "openneuro/ds002680/") and provides methods
    that mirror pathlib / os operations used by the stats parsers.
    """

    def __init__(self, bucket_name, prefix, project=None, client=None):
        self._client = client or storage.Client(project=project)
        self._bucket = self._client.bucket(bucket_name)
        self._prefix = prefix.rstrip("/") + "/"

    @property
    def prefix(self):
        return self._prefix

    def _blob(self, rel_path):
        return self._bucket.blob(self._prefix + rel_path)

    def exists(self, rel_path):
        return self._blob(rel_path).exists()

    def read_text(self, rel_path, encoding="utf-8"):
        blob = self._blob(rel_path)
        return blob.download_as_text(encoding=encoding)

    def read_bytes(self, rel_path, start=None, end=None):
        blob = self._blob(rel_path)
        if start is not None or end is not None:
            return blob.download_as_bytes(start=start, end=end)
        return blob.download_as_bytes()

    def list_blobs(self, suffix=None, rel_prefix=None):
        """List blobs under this dataset, optionally filtered by suffix.

        Yields (relative_path, size_bytes) tuples.
        """
        search_prefix = self._prefix
        if rel_prefix:
            search_prefix = self._prefix + rel_prefix

        for blob in self._client.list_blobs(
            self._bucket, prefix=search_prefix,
            fields="items(name,size),nextPageToken",
        ):
            rel = blob.name[len(self._prefix):]
            if suffix and not rel.endswith(suffix):
                continue
            yield rel, blob.size

    def list_all_blobs(self):
        """List every blob under this dataset. Yields (relative_path, size)."""
        for blob in self._client.list_blobs(
            self._bucket, prefix=self._prefix,
            fields="items(name,size),nextPageToken",
        ):
            yield blob.name[len(self._prefix):], blob.size

    def list_subject_dirs(self):
        """List subject directory prefixes (e.g. 'sub-001/')."""
        subjects = set()
        iterator = self._client.list_blobs(
            self._bucket, prefix=self._prefix + "sub-", delimiter="/"
        )
        for _ in iterator:
            pass
        for pfx in iterator.prefixes:
            name = pfx[len(self._prefix):].rstrip("/")
            subjects.add(name)
        return sorted(subjects)

    def download_to_file(self, rel_path, local_path):
        blob = self._blob(rel_path)
        blob.download_to_filename(local_path)

    def upload_from_file(self, rel_path, local_path):
        blob = self._blob(rel_path)
        blob.upload_from_filename(local_path)

    def upload_text(self, rel_path, content, content_type="text/plain"):
        blob = self._blob(rel_path)
        blob.upload_from_string(content, content_type=content_type)
