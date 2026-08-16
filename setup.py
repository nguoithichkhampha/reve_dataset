"""Setup file for Dataflow worker dependency packaging."""

from setuptools import setup, find_packages

setup(
    name="reve-eeg-pipelines",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-cloud-storage>=2.9.0",
        "mne>=1.6.0",
        "numpy>=1.24.0",
        "h5py>=3.8.0",
    ],
)
