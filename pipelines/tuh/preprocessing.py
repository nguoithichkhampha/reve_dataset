"""TUH EEG preprocessing DoFn — download from GCS, process with MNE, save as HDF5.

Preprocessing steps (in order):
  1. Clean channel names: strip 'EEG ' prefix and '-REF'/'-LE' suffix
  2. Drop non-EEG channels (EKG, EOG, EMG, respiration, photic, markers)
  3. Set standard 10-20 montage
  4. Bandpass filter (0.5-45 Hz, FIR)
  5. Notch filter (60 Hz + harmonics — TUH is US hospital data)
  6. Bad channel detection (RMS > 3 SD from median) and interpolation
  7. Selective re-reference to average
     Only re-references linked-ears montages (02_tcp_le, 04_tcp_le_a).
     Keeps as-is: average-referenced montages (01_tcp_ar, 03_tcp_ar_a).
  8. Resample to target frequency (optional, via --target-sfreq, polyphase)
  9. Z-score normalization per channel + clip at ±15 SD

Two DoFns:
  - TUHPreprocessEEGFn: processes one EDF recording, saves temp HDF5 to GCS,
    yields (group_id, metadata) for downstream grouping.
  - TUHMergeGroupHDF5Fn: merges all recordings per group into one HDF5,
    writes a per-recording manifest for dataloader filtering.

Output: one HDF5 per TUH group (000-149), structured for training dataloader
filtering by n_channels, patient, and session.

  tuh_{group_id}_preprocessed.h5
    attrs: group_id, n_patients, n_recordings, montages[], sfreqs[]
    /{patient_id}/
      /{session}/
        /{token}/
          data: (n_channels, n_samples) float32
          attrs: sfreq, duration_s, n_channels, n_samples, montage, reference
          channel_names: string dataset
          preprocessing/
            attrs: bandpass_low, bandpass_high, reference, z_normalized
            original_sfreq, resampled_to (if resampled)
            bad_channels_detected: string dataset

  manifest_recordings.csv — one row per recording with columns:
    group_id, h5_file, h5_path, patient_id, session, montage, token,
    n_channels, sfreq, n_samples, duration_s
"""

import logging
import os
import re
import shutil
import tempfile
import time

import apache_beam as beam
import h5py
import mne
import numpy as np

from pipelines import BANDPASS_LOW, BANDPASS_HIGH, CLIP_STD, H5_CHUNK_SECONDS
from pipelines.gcs_fs import GCSDatasetFS
from pipelines.tuh.file_groups import TUHFileGroup

logger = logging.getLogger(__name__)

NON_EEG_PATTERNS = re.compile(
    r"^(EKG\d?|ECG[_ ]EKG|LOC|ROC|RLC|LUC|EMG|RESP\d?|RESP ABDOMEN|"
    r"PHOTIC|PHOTIC PH|IBI|BURSTS|SUPPR|DC\d+|"
    r"EEG EKG\d?|EEG LOC|EEG ROC|EEG RLC|EEG LUC|"
    r"EEG RESP\d?|PHOTIC-|PULSE|EDF ANNOTATIONS|"
    r"EVENTS/?MARKERS|CO2WAVE|ETCO2|SPO2|"
    r"EEG MARK\d+|EEG E$|EEG X\d+|EEG 1X10_)",
    re.IGNORECASE,
)

MONTAGE_REF_MAP = {
    "01_tcp_ar": "average",
    "02_tcp_le": "linked ears",
    "03_tcp_ar_a": "average",
    "04_tcp_le_a": "linked ears",
}

TUH_POWERLINE_FREQ = 60.0

_TUH_TO_MONTAGE_CASE = {
    "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "OZ": "Oz",
    "FCZ": "FCz", "CPZ": "CPz", "AFZ": "AFz", "POZ": "POz",
}


def _clean_channel_name(ch_name):
    """Strip TUH channel prefixes/suffixes and fix casing for montage matching.

    'EEG FP1-REF' -> 'Fp1'
    'EEG CZ-LE'   -> 'Cz'
    'EMG-REF'     -> 'EMG'
    """
    name = ch_name.strip()
    if name.startswith("EEG "):
        name = name[4:]
    name = re.sub(r"-(REF|LE|AR)$", "", name, flags=re.IGNORECASE)
    name = name.strip()
    return _TUH_TO_MONTAGE_CASE.get(name.upper(), name)


def _is_eeg_channel(ch_name):
    """Return True if this looks like an EEG channel (not EKG, EOG, EMG, etc.)."""
    if NON_EEG_PATTERNS.match(ch_name):
        return False
    clean = _clean_channel_name(ch_name)
    non_eeg_clean = {
        "EKG", "EKG1", "EKG2", "LOC", "ROC", "RLC", "LUC",
        "EMG", "RESP", "RESP1", "RESP2", "RESP ABDOMEN",
        "PHOTIC", "PHOTIC PH", "IBI", "BURSTS", "SUPPR",
        "PULSE", "PULSE RATE", "E", "ECG EKG",
        "CO2WAVE", "ETCO2", "SPO2", "EVENTS/MARKERS",
    }
    if clean.upper() in non_eeg_clean:
        return False
    if re.match(r"^DC\d+$", clean, re.IGNORECASE):
        return False
    if re.match(r"^\d+$", clean):
        return False
    if re.match(r"^MARK\d+$", clean, re.IGNORECASE):
        return False
    if re.match(r"^X\d+$", clean, re.IGNORECASE):
        return False
    if re.match(r"^1X10_", clean, re.IGNORECASE):
        return False
    if "-" in clean and re.match(r"^[A-Z][A-Za-z0-9]+-[A-Z][A-Za-z0-9]+$", clean):
        return False
    return True


class TUHPreprocessEEGFn(beam.DoFn):
    """Download, preprocess, and upload one TUH EDF recording.

    Yields (group_id, metadata_dict) tuples for downstream grouping.
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
        fg = TUHFileGroup.from_dict(file_group_dict)
        t0 = time.time()
        tmp_dir = tempfile.mkdtemp(prefix="tuh_preproc_")
        meta = {
            "group": fg.group,
            "patient_id": fg.patient_id,
            "session": fg.session,
            "montage": fg.montage,
            "token": fg.token,
            "input_blob": fg.blob_path,
            "status": "failed",
        }

        try:
            fs = GCSDatasetFS(self._bucket_name, self._prefix, project=self._project)

            local_path = os.path.join(tmp_dir, os.path.basename(fg.blob_path))
            fs.download_to_file(fg.blob_path, local_path)

            raw = mne.io.read_raw_edf(local_path, preload=True)

            eeg_picks = [ch for ch in raw.ch_names if _is_eeg_channel(ch)]
            if not eeg_picks:
                meta["error"] = "No EEG channels found after filtering"
                yield beam.pvalue.TaggedOutput("failed", meta)
                return

            non_eeg = [ch for ch in raw.ch_names if ch not in eeg_picks]
            if non_eeg:
                raw.drop_channels(non_eeg)

            duration_s = raw.n_times / raw.info["sfreq"]
            if duration_s < H5_CHUNK_SECONDS:
                meta["error"] = f"Recording too short ({duration_s:.1f}s < {H5_CHUNK_SECONDS}s)"
                yield beam.pvalue.TaggedOutput("failed", meta)
                return

            rename_map = {}
            for ch in raw.ch_names:
                clean = _clean_channel_name(ch)
                if clean != ch:
                    rename_map[ch] = clean
            if rename_map:
                raw.rename_channels(rename_map)

            preproc_meta = self._preprocess(raw, fg.montage)
            meta.update(preproc_meta)

            stem = f"{fg.patient_id}_{fg.session}_{fg.token}"
            temp_blob = f"{fg.group}/{stem}.h5"
            h5_local = os.path.join(tmp_dir, "output.h5")
            self._save_hdf5(raw, h5_local, preproc_meta, fg.montage)

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
            logger.error("Failed %s/%s: %s", fg.group, fg.blob_path, e)
            meta["error"] = str(e)
            meta["processing_time_s"] = round(time.time() - t0, 1)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if meta["status"] == "success":
            yield (fg.group, meta)
        else:
            yield beam.pvalue.TaggedOutput("failed", meta)

    MONTAGE_CANDIDATES = [
        "standard_1020",
        "standard_1005",
    ]

    @staticmethod
    def _set_montage(raw):
        """Try standard montages, pick the one matching the most channels."""
        ch_set = set(raw.ch_names)
        best_montage = None
        best_matched = 0

        for name in TUHPreprocessEEGFn.MONTAGE_CANDIDATES:
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

    def _preprocess(self, raw, montage_dir):
        reference = MONTAGE_REF_MAP.get(montage_dir, "unknown")
        meta = {"bad_channels": [], "reference": "average"}

        self._set_montage(raw)

        raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, fir_design="firwin")

        freqs = [TUH_POWERLINE_FREQ]
        if TUH_POWERLINE_FREQ * 2 <= raw.info["sfreq"] / 2:
            freqs.append(TUH_POWERLINE_FREQ * 2)
        raw.notch_filter(freqs)

        rms = np.sqrt(np.mean(raw._data ** 2, axis=1))
        median_rms = np.median(rms)
        bad_mask = np.abs(rms - median_rms) > 3 * np.std(rms)
        bad_channels = [raw.ch_names[i] for i, is_bad in enumerate(bad_mask) if is_bad]
        del rms

        if bad_channels and len(bad_channels) < len(raw.ch_names) * 0.3:
            pos = raw._get_channel_positions()
            has_pos = {
                raw.ch_names[i]
                for i in range(len(pos))
                if not np.any(np.isnan(pos[i]))
            }
            interpolable = [ch for ch in bad_channels if ch in has_pos]
            no_pos = [ch for ch in raw.ch_names if ch not in has_pos]

            if interpolable:
                raw.info["bads"] = interpolable
                try:
                    raw.interpolate_bads(reset_bads=True, exclude=no_pos)
                except Exception:
                    raw.info["bads"] = []
            else:
                raw.info["bads"] = []
        meta["bad_channels"] = bad_channels

        already_avg = reference == "average"
        if already_avg:
            meta["reference"] = "average"
            meta["original_reference"] = montage_dir
        else:
            raw.set_eeg_reference("average", projection=False)
            meta["reference"] = "average"
            meta["original_reference"] = montage_dir

        if self._target_sfreq and raw.info["sfreq"] != self._target_sfreq:
            meta["original_sfreq"] = raw.info["sfreq"]
            raw.resample(self._target_sfreq, method="polyphase")
            meta["resampled_to"] = self._target_sfreq

        data = raw._data
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std < 1e-10] = 1.0
        data -= mean
        data /= std
        del mean, std
        np.clip(data, -CLIP_STD, CLIP_STD, out=data)
        meta["z_normalized"] = True
        meta["clip_std"] = CLIP_STD

        return meta

    def _save_hdf5(self, raw, h5_path, preproc_meta, montage_dir):
        data = raw._data.astype(np.float32)
        raw._data = None
        chunk_samples = int(raw.info["sfreq"] * H5_CHUNK_SECONDS)
        chunk_samples = min(chunk_samples, data.shape[1])
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data, chunks=(data.shape[0], chunk_samples))

            f.attrs["sfreq"] = raw.info["sfreq"]
            f.attrs["n_channels"] = data.shape[0]
            f.attrs["n_samples"] = data.shape[1]
            f.attrs["duration_s"] = data.shape[1] / raw.info["sfreq"]
            f.attrs["reference"] = preproc_meta.get("reference", "average")
            f.attrs["montage"] = montage_dir

            ch_names = [n.encode("utf-8") for n in raw.ch_names]
            f.create_dataset("channel_names", data=ch_names)

            preproc_group = f.create_group("preprocessing")
            preproc_group.attrs["bandpass_low"] = BANDPASS_LOW
            preproc_group.attrs["bandpass_high"] = BANDPASS_HIGH
            preproc_group.attrs["notch_freq"] = TUH_POWERLINE_FREQ
            preproc_group.attrs["reference"] = preproc_meta.get("reference", "average")
            preproc_group.attrs["z_normalized"] = preproc_meta.get("z_normalized", False)

            if preproc_meta.get("bad_channels"):
                bad_ch = [c.encode("utf-8") for c in preproc_meta["bad_channels"]]
                preproc_group.create_dataset("bad_channels_detected", data=bad_ch)

            if preproc_meta.get("resampled_to"):
                preproc_group.attrs["original_sfreq"] = preproc_meta["original_sfreq"]
                preproc_group.attrs["resampled_to"] = preproc_meta["resampled_to"]


class TUHMergeGroupHDF5Fn(beam.DoFn):
    """Merge all preprocessed recordings for one TUH group into a single HDF5.

    Input: (group_id, [list of recording metadata dicts])
    Output: summary dict, tagged "manifest" outputs for CSV rows.

    HDF5 structure:
      attrs: group_id, n_patients, n_recordings, montages[], sfreqs[]
      /{patient_id}/{session}/{token}/
        data, channel_names, preprocessing/...
    """

    def __init__(self, bucket_name, output_prefix, project=None):
        self._bucket_name = bucket_name
        self._output_prefix = output_prefix
        self._project = project

    def process(self, element):
        group_id, recording_metas = element
        recording_metas = list(recording_metas)

        if not recording_metas:
            return

        logger.info("Merging %d recordings for group %s", len(recording_metas), group_id)

        tmp_dir = tempfile.mkdtemp(prefix="tuh_merge_")
        temp_prefix = self._output_prefix.rstrip("/") + "_temp/"
        temp_fs = GCSDatasetFS(self._bucket_name, temp_prefix, project=self._project)

        try:
            merged_path = os.path.join(tmp_dir, f"tuh_{group_id}_preprocessed.h5")

            all_patients = set()
            all_montages = set()
            channel_counts = set()
            sfreqs = set()

            with h5py.File(merged_path, "w") as merged:
                for meta in recording_metas:
                    patient_id = meta.get("patient_id", "unknown")
                    session = meta.get("session", "s001")
                    token = meta.get("token", "t000")
                    montage = meta.get("montage", "")

                    all_patients.add(patient_id)
                    all_montages.add(montage)

                    group_path = f"{patient_id}/{session}/{token}"

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

                        n_ch = int(src.attrs.get("n_channels", 0))
                        sr = float(src.attrs.get("sfreq", 0))
                        channel_counts.add(n_ch)
                        sfreqs.add(sr)

                        meta["n_channels"] = n_ch
                        meta["sfreq"] = sr
                        meta["n_samples"] = int(src.attrs.get("n_samples", 0))
                        meta["duration_s"] = float(src.attrs.get("duration_s", 0))

                    os.remove(local_h5)

                merged.attrs["group_id"] = group_id
                merged.attrs["n_recordings"] = len(recording_metas)
                merged.attrs["n_patients"] = len(all_patients)
                merged.attrs["montages"] = sorted(all_montages)
                merged.attrs["patients"] = sorted(all_patients)

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
            output_blob = f"tuh_{group_id}_preprocessed.h5"
            out_fs.upload_from_file(output_blob, merged_path)

            self._cleanup_temp_blobs(temp_fs, recording_metas)

            output_path = f"gs://{self._bucket_name}/{self._output_prefix}{output_blob}"
            file_size = os.path.getsize(merged_path)
            logger.info(
                "Merged group %s: %d recordings -> %s (%.1f MB)",
                group_id, len(recording_metas), output_path,
                file_size / 1024 / 1024,
            )

            for meta in recording_metas:
                yield beam.pvalue.TaggedOutput("manifest", {
                    "group_id": group_id,
                    "h5_file": output_blob,
                    "h5_path": f"{meta.get('patient_id', 'unknown')}/{meta.get('session', 's001')}/{meta.get('token', 't000')}",
                    "patient_id": meta.get("patient_id", ""),
                    "session": meta.get("session", ""),
                    "montage": meta.get("montage", ""),
                    "token": meta.get("token", ""),
                    "n_channels": meta.get("n_channels", 0),
                    "sfreq": meta.get("sfreq", 0),
                    "n_samples": meta.get("n_samples", 0),
                    "duration_s": meta.get("duration_s", 0),
                })

            yield {
                "group_id": group_id,
                "output_path": output_path,
                "n_recordings": len(recording_metas),
                "n_patients": len(all_patients),
                "montages": sorted(all_montages),
                "output_size_bytes": file_size,
            }

        except Exception as e:
            logger.error("Merge failed for group %s: %s", group_id, e)
            yield {
                "group_id": group_id,
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
