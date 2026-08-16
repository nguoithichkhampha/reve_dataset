"""EEG preprocessing DoFn — download from GCS, process with MNE, save as HDF5.

Preprocessing steps (in order):
  1. Auto-select montage by best channel match:
     standard_1020, standard_1005, GSN-HydroCel-{32,64,65,128,129,256,257}, EGI_256
  2. Bandpass filter (0.1-100 Hz, FIR)
  3. Notch filter (50/60 Hz + harmonics, from sidecar PowerLineFrequency)
  4. Drop flat reference channel if present in data
     (channel name parsed from sidecar EEGReference, verified flat by RMS < 1% median)
  5. Bad channel detection (RMS > 3 SD from median) and interpolation
  6. Re-reference to average
     (skipped if sidecar EEGReference indicates already average-referenced,
      e.g. "average", "common", "Cz; common")
  7. Z-score normalization per channel (mean=0, std=1 computed across recording),
     then clip values exceeding ±15 standard deviations
  8. Resample to target frequency (optional, via --target-sfreq)

Two DoFns:
  - PreprocessEEGFn: processes one recording, saves temp HDF5 to GCS,
    yields (dataset_id, metadata) for downstream grouping.
  - MergeDatasetHDF5Fn: groups all recordings per dataset into one HDF5,
    writes a per-recording manifest for dataloader filtering.

Output: one HDF5 per dataset, structured for training dataloader filtering
by n_channels, task, and subject.

  {dataset_id}_preprocessed.h5
    attrs: dataset_id, n_channels, sfreq, tasks[], subjects[], n_subjects,
           n_recordings
    channel_names: dataset-level reference channel list
    /{subject}/
      /{task}/
        /{run}/
          data: (n_channels, n_samples) float32
          attrs: sfreq, duration_s, n_channels, n_samples, session, reference
          channel_names: string dataset
          preprocessing/
            attrs: bandpass_low, bandpass_high, reference, z_normalized
            original_sfreq, resampled_to (if resampled)
            bad_channels_detected: string dataset

  manifest_recordings.csv — one row per recording with columns:
    dataset_id, h5_file, h5_path, subject, session, task, run,
    n_channels, sfreq, n_samples, duration_s
"""

import logging
import os
import shutil
import tempfile
import time

import apache_beam as beam
import h5py
import mne
import numpy as np

from pipelines.gcs_fs import GCSDatasetFS
from pipelines.eeg.file_groups import EEGFileGroup
from pipelines.stats.parsers import parse_eeg_json

logger = logging.getLogger(__name__)

MNE_READERS = {
    "brainvision": mne.io.read_raw_brainvision,
    "eeglab": mne.io.read_raw_eeglab,
    "biosemi": mne.io.read_raw_bdf,
    "edf": mne.io.read_raw_edf,
    "mne": mne.io.read_raw_fif,
}


class PreprocessEEGFn(beam.DoFn):
    """Download, preprocess, and upload one EEG recording to a temp GCS path.

    Yields (dataset_id, metadata_dict) tuples for downstream grouping.
    The temp HDF5 is stored at {output_prefix}_temp/{dataset_id}/{stem}.h5
    and will be merged into one file per dataset by MergeDatasetHDF5Fn.
    """

    def __init__(self, bucket_name, prefix, output_prefix, project=None, target_sfreq=None):
        self._bucket_name = bucket_name
        self._prefix = prefix
        self._output_prefix = output_prefix
        self._project = project
        self._target_sfreq = target_sfreq

    def setup(self):
        mne.set_log_level("WARNING")

    def process(self, file_group_dict):
        fg = EEGFileGroup.from_dict(file_group_dict)
        t0 = time.time()
        tmp_dir = tempfile.mkdtemp(prefix="eeg_preproc_")
        meta = {
            "dataset_id": fg.dataset_id,
            "subject": fg.subject,
            "session": fg.session,
            "task": fg.task,
            "run": fg.run or "1",
            "format": fg.format,
            "input_blob": fg.primary_blob,
            "status": "failed",
        }

        try:
            ds_prefix = f"{self._prefix}{fg.dataset_id}"
            fs = GCSDatasetFS(self._bucket_name, ds_prefix, project=self._project)

            all_blobs = [fg.primary_blob] + fg.aux_blobs
            for blob_path in all_blobs:
                local_path = os.path.join(tmp_dir, blob_path.replace("/", os.sep))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                fs.download_to_file(blob_path, local_path)

            primary_local = os.path.join(
                tmp_dir, fg.primary_blob.replace("/", os.sep)
            )

            sidecar_params = self._get_sidecar_params(fs, fg)

            reader = MNE_READERS.get(fg.format)
            if not reader:
                meta["error"] = f"Unknown format: {fg.format}"
                yield beam.pvalue.TaggedOutput("failed", meta)
                return

            raw = reader(primary_local, preload=True)
            raw.pick_types(eeg=True, exclude=[])

            if raw.get_data().nbytes == 0:
                meta["error"] = "No EEG channels found"
                yield beam.pvalue.TaggedOutput("failed", meta)
                return

            preproc_meta = self._preprocess(raw, sidecar_params)
            meta.update(preproc_meta)

            stem = os.path.splitext(fg.primary_blob.rsplit("/", 1)[-1])[0]
            temp_blob = f"{fg.dataset_id}/{stem}.h5"
            h5_local = os.path.join(tmp_dir, "output.h5")
            self._save_hdf5(raw, h5_local, preproc_meta)

            temp_prefix = self._output_prefix.rstrip("/") + "_temp/"
            temp_fs = GCSDatasetFS(self._bucket_name, temp_prefix, project=self._project)
            temp_fs.upload_from_file(temp_blob, h5_local)

            meta["status"] = "success"
            meta["temp_blob"] = temp_blob
            meta["n_channels"] = len(raw.ch_names)
            meta["sfreq"] = raw.info["sfreq"]
            meta["channel_names"] = raw.ch_names
            meta["processing_time_s"] = round(time.time() - t0, 1)
            meta["output_size_bytes"] = os.path.getsize(h5_local)

        except Exception as e:
            logger.error("Failed %s/%s: %s", fg.dataset_id, fg.primary_blob, e)
            meta["error"] = str(e)
            meta["processing_time_s"] = round(time.time() - t0, 1)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if meta["status"] == "success":
            yield (fg.dataset_id, meta)
        else:
            yield beam.pvalue.TaggedOutput("failed", meta)

    def _get_sidecar_params(self, fs, fg):
        """Read powerline frequency and EEG reference from sidecar JSON."""
        result = {"powerline_freq": 50.0, "reference": None}

        sources = list(fg.sidecar_blobs)
        try:
            sources.extend(path for path, _ in fs.list_blobs(suffix="_eeg.json"))
        except Exception:
            pass

        for path in sources:
            try:
                text = fs.read_text(path)
                params = parse_eeg_json(text)
                if "powerline_freq" in params and result["powerline_freq"] == 50.0:
                    result["powerline_freq"] = params["powerline_freq"]
                if "reference" in params and result["reference"] is None:
                    result["reference"] = params["reference"]
            except Exception:
                continue

        return result

    MONTAGE_CANDIDATES = [
        "standard_1020",
        "standard_1005",
        "GSN-HydroCel-129",
        "GSN-HydroCel-128",
        "GSN-HydroCel-65_1.0",
        "GSN-HydroCel-64_1.0",
        "GSN-HydroCel-32",
        "GSN-HydroCel-257",
        "GSN-HydroCel-256",
        "EGI_256",
    ]

    @staticmethod
    def _set_montage(raw):
        """Try montages in order, pick the one matching the most channels."""
        ch_set = set(raw.ch_names)
        best_montage = None
        best_matched = 0

        for name in PreprocessEEGFn.MONTAGE_CANDIDATES:
            try:
                montage = mne.channels.make_standard_montage(name)
                matched = len(ch_set & set(montage.ch_names))
                if matched > best_matched:
                    best_matched = matched
                    best_montage = montage
            except Exception:
                continue

        if best_montage and best_matched > 0:
            raw.set_montage(best_montage, on_missing="ignore")

    def _preprocess(self, raw, sidecar_params):
        powerline_freq = sidecar_params.get("powerline_freq", 50.0)
        dataset_ref = sidecar_params.get("reference")
        meta = {"bad_channels": [], "reference": "average"}

        self._set_montage(raw)

        raw.filter(l_freq=0.1, h_freq=100.0, fir_design="firwin")

        if powerline_freq and powerline_freq > 0:
            freqs = [powerline_freq]
            if powerline_freq * 2 <= raw.info["sfreq"] / 2:
                freqs.append(powerline_freq * 2)
            raw.notch_filter(freqs)

        ref_channel = self._find_ref_channel_in_data(raw, dataset_ref)
        if ref_channel:
            raw.drop_channels([ref_channel])
            meta["dropped_ref_channel"] = ref_channel
            logger.info("Dropped original reference channel '%s' from data", ref_channel)

        ch_data = raw.get_data()
        rms = np.sqrt(np.mean(ch_data ** 2, axis=1))
        median_rms = np.median(rms)
        std_rms = np.std(rms)
        bad_mask = np.abs(rms - median_rms) > 3 * std_rms
        bad_channels = [raw.ch_names[i] for i, is_bad in enumerate(bad_mask) if is_bad]

        if bad_channels and len(bad_channels) < len(raw.ch_names) * 0.3:
            raw.info["bads"] = bad_channels
            try:
                raw.interpolate_bads(reset_bads=True)
            except Exception:
                raw.info["bads"] = []
        meta["bad_channels"] = bad_channels

        avg_keywords = {"average", "average reference", "common average",
                        "common average reference", "car", "common"}
        ref_parts = {p.strip().lower() for p in dataset_ref.split(";")} if dataset_ref else set()
        already_avg = bool(ref_parts & avg_keywords)
        if already_avg:
            meta["reference"] = dataset_ref
            logger.info("Skipping re-reference: already '%s'", dataset_ref)
        else:
            raw.set_eeg_reference("average", projection=False)
            meta["reference"] = "average"
            if dataset_ref:
                meta["original_reference"] = dataset_ref

        data = raw.get_data()
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std < 1e-10] = 1.0
        normalized = (data - mean) / std
        np.clip(normalized, -15, 15, out=normalized)
        raw._data = normalized
        meta["z_normalized"] = True
        meta["clip_std"] = 15

        if self._target_sfreq and raw.info["sfreq"] != self._target_sfreq:
            meta["original_sfreq"] = raw.info["sfreq"]
            raw.resample(self._target_sfreq)
            meta["resampled_to"] = self._target_sfreq

        return meta

    SKIP_REF_TOKENS = {
        "average", "average reference", "common average",
        "common average reference", "car", "common",
        "cms/drl", "cms", "drl",
        "n/a", "na", "", "linked mastoids", "linked ears",
        "mastoids", "contralateral", "contralateral mastoids",
        "infinity", "rest", "abr", "ref",
        "reference", "placed", "on", "between",
    }

    @staticmethod
    def _parse_ref_channel_candidates(dataset_ref):
        """Extract possible EEG channel names from a compound EEGReference string.

        Handles real-world formats from OpenNeuro datasets:
          "Cz"                                    -> [cz]
          "Cz; common"                            -> [cz]
          "Cz; FCz"                               -> [cz, fcz]
          "FCz, re-referenced to average"         -> [fcz]
          "FCz (online), average (offline)"       -> [fcz]
          "ABR reference placed on FCz"           -> [fcz]
          "between_Cz_and_CPz"                    -> [cz, cpz]
          "Contralateral mastoids (TP9, TP10)"    -> [tp9, tp10]
          "placed on FCz"                         -> [fcz]
        """
        import re
        ref = dataset_ref.strip()

        paren_channels = re.findall(r"\(([^)]+)\)", ref)
        paren_candidates = []
        for content in paren_channels:
            for token in re.split(r"[,;\s]+", content):
                token = token.strip().lower()
                if token and token not in PreprocessEEGFn.SKIP_REF_TOKENS:
                    paren_candidates.append(token)

        ref_clean = re.sub(r"\(.*?\)", "", ref).strip()

        parts = re.split(r"[;,]\s*", ref_clean)

        candidates = []
        for part in parts:
            part = re.sub(
                r"\b(re-?referenced?\s*(to)?|online|offline|initially|then)\b",
                "", part, flags=re.IGNORECASE,
            ).strip()

            tokens = re.split(r"[\s_]+and[\s_]+|[\s_]+", part)
            for token in tokens:
                token = token.strip().lower()
                if token and token not in PreprocessEEGFn.SKIP_REF_TOKENS:
                    candidates.append(token)

        candidates.extend(paren_candidates)
        return candidates

    @staticmethod
    def _find_ref_channel_in_data(raw, dataset_ref):
        """Check if the original reference channel is present and flat.

        Parses compound reference strings, matches against channel names,
        and verifies the channel is flat (RMS < 1% of median) before
        recommending removal.
        """
        if not dataset_ref:
            return None

        ref_lower = dataset_ref.lower().strip()
        if ref_lower in PreprocessEEGFn.SKIP_REF_TOKENS:
            return None

        candidates = PreprocessEEGFn._parse_ref_channel_candidates(dataset_ref)
        if not candidates:
            return None

        ch_names_lower = {ch.lower(): ch for ch in raw.ch_names}
        all_rms = np.sqrt(np.mean(raw.get_data() ** 2, axis=1))
        median_rms = np.median(all_rms)

        for candidate in candidates:
            if candidate in ch_names_lower:
                ch_name = ch_names_lower[candidate]
                idx = raw.ch_names.index(ch_name)
                ch_data = raw.get_data(picks=[idx])
                rms = np.sqrt(np.mean(ch_data ** 2))
                if median_rms > 0 and rms < median_rms * 0.01:
                    return ch_name
        return None

    def _save_hdf5(self, raw, h5_path, preproc_meta):
        data = raw.get_data().astype(np.float32)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data, compression="gzip", compression_opts=4)

            f.attrs["sfreq"] = raw.info["sfreq"]
            f.attrs["n_channels"] = data.shape[0]
            f.attrs["n_samples"] = data.shape[1]
            f.attrs["duration_s"] = data.shape[1] / raw.info["sfreq"]
            f.attrs["reference"] = preproc_meta.get("reference", "average")

            ch_names = [n.encode("utf-8") for n in raw.ch_names]
            f.create_dataset("channel_names", data=ch_names)

            preproc_group = f.create_group("preprocessing")
            preproc_group.attrs["bandpass_low"] = 0.1
            preproc_group.attrs["bandpass_high"] = 100.0
            preproc_group.attrs["reference"] = preproc_meta.get("reference", "average")
            preproc_group.attrs["z_normalized"] = preproc_meta.get("z_normalized", False)

            if preproc_meta.get("bad_channels"):
                bad_ch = [c.encode("utf-8") for c in preproc_meta["bad_channels"]]
                preproc_group.create_dataset("bad_channels_detected", data=bad_ch)

            if preproc_meta.get("resampled_to"):
                preproc_group.attrs["original_sfreq"] = preproc_meta["original_sfreq"]
                preproc_group.attrs["resampled_to"] = preproc_meta["resampled_to"]


class MergeDatasetHDF5Fn(beam.DoFn):
    """Merge all preprocessed recordings for one dataset into a single HDF5.

    Input: (dataset_id, [list of recording metadata dicts])
    Output: summary dict

    HDF5 structure:
      attrs: dataset_id, n_channels, sfreq, tasks[], n_subjects, n_recordings
      /{subject}/{task}/{run}/
        data, channel_names, preprocessing/...
    """

    def __init__(self, bucket_name, output_prefix, project=None):
        self._bucket_name = bucket_name
        self._output_prefix = output_prefix
        self._project = project

    def process(self, element):
        dataset_id, recording_metas = element
        recording_metas = list(recording_metas)

        if not recording_metas:
            return

        logger.info(
            "Merging %d recordings for %s", len(recording_metas), dataset_id
        )

        tmp_dir = tempfile.mkdtemp(prefix="eeg_merge_")
        temp_prefix = self._output_prefix.rstrip("/") + "_temp/"
        temp_fs = GCSDatasetFS(self._bucket_name, temp_prefix, project=self._project)

        try:
            merged_path = os.path.join(tmp_dir, f"{dataset_id}_preprocessed.h5")

            all_tasks = set()
            all_subjects = set()
            channel_counts = set()
            sfreqs = set()

            with h5py.File(merged_path, "w") as merged:
                for meta in recording_metas:
                    subject = meta.get("subject", "unknown")
                    task = meta.get("task", "unknown")
                    run = meta.get("run", "1")
                    session = meta.get("session", "")

                    all_subjects.add(subject)
                    all_tasks.add(task)

                    group_path = f"{subject}/{task}/{run}"

                    temp_blob = meta["temp_blob"]
                    local_h5 = os.path.join(tmp_dir, temp_blob.replace("/", "_"))
                    temp_fs.download_to_file(temp_blob, local_h5)

                    with h5py.File(local_h5, "r") as src:
                        grp = merged.create_group(group_path)

                        src.copy("data", grp)
                        src.copy("channel_names", grp)
                        if "preprocessing" in src:
                            src.copy("preprocessing", grp)

                        for attr_name, attr_val in src.attrs.items():
                            grp.attrs[attr_name] = attr_val

                        if session:
                            grp.attrs["session"] = session

                        n_ch = int(src.attrs.get("n_channels", 0))
                        sr = float(src.attrs.get("sfreq", 0))
                        channel_counts.add(n_ch)
                        sfreqs.add(sr)

                        meta["n_channels"] = n_ch
                        meta["sfreq"] = sr
                        meta["n_samples"] = int(src.attrs.get("n_samples", 0))
                        meta["duration_s"] = float(src.attrs.get("duration_s", 0))

                    os.remove(local_h5)

                merged.attrs["dataset_id"] = dataset_id
                merged.attrs["n_recordings"] = len(recording_metas)
                merged.attrs["n_subjects"] = len(all_subjects)
                merged.attrs["tasks"] = sorted(all_tasks)
                merged.attrs["subjects"] = sorted(all_subjects)

                if len(channel_counts) == 1:
                    merged.attrs["n_channels"] = channel_counts.pop()
                else:
                    merged.attrs["n_channels"] = sorted(channel_counts)

                if len(sfreqs) == 1:
                    merged.attrs["sfreq"] = sfreqs.pop()
                else:
                    merged.attrs["sfreq"] = sorted(sfreqs)

                if recording_metas and recording_metas[0].get("channel_names"):
                    ch = [n.encode("utf-8") for n in recording_metas[0]["channel_names"]]
                    merged.create_dataset("channel_names", data=ch)

            out_fs = GCSDatasetFS(
                self._bucket_name, self._output_prefix, project=self._project
            )
            output_blob = f"{dataset_id}_preprocessed.h5"
            out_fs.upload_from_file(output_blob, merged_path)

            self._cleanup_temp_blobs(temp_fs, recording_metas)

            output_path = f"gs://{self._bucket_name}/{self._output_prefix}{output_blob}"
            file_size = os.path.getsize(merged_path)
            logger.info(
                "Merged %s: %d recordings -> %s (%.1f MB)",
                dataset_id, len(recording_metas), output_path,
                file_size / 1024 / 1024,
            )

            for meta in recording_metas:
                yield beam.pvalue.TaggedOutput("manifest", {
                    "dataset_id": dataset_id,
                    "h5_file": output_blob,
                    "h5_path": f"{meta.get('subject', 'unknown')}/{meta.get('task', 'unknown')}/{meta.get('run', '1')}",
                    "subject": meta.get("subject", ""),
                    "session": meta.get("session", ""),
                    "task": meta.get("task", ""),
                    "run": meta.get("run", "1"),
                    "n_channels": meta.get("n_channels", 0),
                    "sfreq": meta.get("sfreq", 0),
                    "n_samples": meta.get("n_samples", 0),
                    "duration_s": meta.get("duration_s", 0),
                })

            yield {
                "dataset_id": dataset_id,
                "output_path": output_path,
                "n_recordings": len(recording_metas),
                "n_subjects": len(all_subjects),
                "tasks": sorted(all_tasks),
                "output_size_bytes": file_size,
            }

        except Exception as e:
            logger.error("Merge failed for %s: %s", dataset_id, e)
            yield {
                "dataset_id": dataset_id,
                "status": "merge_failed",
                "error": str(e),
            }

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _cleanup_temp_blobs(self, temp_fs, recording_metas):
        from google.cloud import storage
        client = storage.Client(project=self._project)
        bucket = client.bucket(self._bucket_name)
        temp_prefix = self._output_prefix.rstrip("/") + "_temp/"
        for meta in recording_metas:
            blob_name = temp_prefix + meta["temp_blob"]
            bucket.blob(blob_name).delete()
