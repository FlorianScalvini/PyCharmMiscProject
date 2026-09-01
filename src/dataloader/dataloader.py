"""
PyTorch Lightning data module for spatio-temporal longitudinal brain MRI.

Reads train and validation subject lists from JSON manifests, normalises
acquisition ages to the ``[0, 1]`` interval defined by ``[t0, tn]``, sorts
sessions chronologically, and exposes standard Lightning DataLoader hooks.

JSON manifest format
--------------------
Each JSON file must follow the structure::

    {
        "subjects": [
            {
                "sessions": [
                    {
                        "image": "relative/path/to/image.nii.gz",
                        "segmentation": "relative/path/to/seg.nii.gz",
                        "age": 85.0
                    },
                    ...
                ]
            },
            ...
        ]
    }

All image paths are interpreted relative to ``root_dir``.

Author : Florian Scalvini
"""

# --- Standard library ---
import os
import glob
import json
import random

# --- Third-party ---
import torch
import pandas as pd
import torchio as tio
import pytorch_lightning as pl

# --- Local ---
from dataloader.dataset import SpatioTemporalDatasetValidation, SpatioTemporalDataset


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────

def split_and_shuffled(
    lst: list,
    ratio: float,
    seed: int | None = None,
) -> tuple[list, list]:
    """Randomly split *lst* into two parts according to *ratio*.

    Parameters
    ----------
    lst : list
        The sequence to split.  The original list is not modified.
    ratio : float
        Fraction of elements placed in the first partition (``0 < ratio < 1``).
    seed : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    first : list
        First ``floor(len(lst) * ratio)`` elements after shuffling.
    second : list
        Remaining elements.
    """
    lst_copy = lst.copy()  # avoid modifying original list
    if seed is not None:
        random.seed(seed)  # for reproducibility
    random.shuffle(lst_copy)
    split_idx = int(len(lst_copy) * ratio)
    return lst_copy[:split_idx], lst_copy[split_idx:]


# ──────────────────────────────────────────────────────────────────────────────
#  Data module
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalSequenceDatamoduleJSON(pl.LightningDataModule):
    """Lightning data module for longitudinal brain MRI sequences.

    Parses two JSON manifests (train / validation), normalises ages, builds
    :class:`~dataloader.dataset.SpatioTemporalDataset` and
    :class:`~dataloader.dataset.SpatioTemporalDatasetValidation` instances on
    demand, and returns single-sample DataLoaders configured for low-memory
    3-D volume loading.

    Parameters
    ----------
    root_dir : str
        Absolute path prepended to all relative image paths in the JSON files.
    json_path : str
        Path to the training manifest JSON, relative to *root_dir*.
    json_path_val : str
        Path to the validation manifest JSON, relative to *root_dir*.
    batch_size : int
        Batch size forwarded to the Lightning trainer (stored but DataLoaders
        always use ``batch_size=1`` for memory reasons).
    seed : int
        Random seed used by :func:`split_and_shuffled`.
    num_workers : int
        Number of DataLoader worker processes.
    t0 : float
        Age at the start of the developmental window — maps to ``0``.
    tn : float
        Age at the end of the developmental window — maps to ``1``.
    size : tuple of int
        Target spatial dimensions ``(D, H, W)`` after resizing.
    crop : tuple of int
        Crop/pad target ``(D, H, W)`` applied before resizing.
    """

    def __init__(
        self,
        root_dir: str,
        json_path: str,
        json_path_val: str,
        batch_size: int,
        seed: int = 42,
        num_workers: int = 4,
        size: tuple[int, int, int] = (192, 224, 192),
        crop: tuple[int, int, int] = (50, 50, 50),

    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        if batch_size != 1:
            raise ValueError(
                "longitudinal sequences have variable lengths; batch_size must be 1"
            )
        self.json_path = os.path.join(root_dir, json_path)
        self.json_path_val = os.path.join(root_dir, json_path_val)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_subjects = None
        self.seed = seed
        self.size = size
        self.crop = crop
        self.transform = tio.transforms.Compose([
            tio.transforms.CropOrPad(crop),
            tio.transforms.Resize(size),
            tio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.05, 99.5)),
        ])
        self.transform_seg = tio.transforms.Compose([
            tio.transforms.CropOrPad(crop),
            tio.transforms.Resize(size),
        ])
        self.data_train: list = []
        self.data_val: list = []
        segmentation_presence: list[bool] = []

        with open(self.json_path, 'r') as f:
            # Parsing the JSON file into a Python dictionary
            data = json.load(f)

        for i in range(len(data['subjects'])):
            subject = []
            for j in range(len(data['subjects'][i]['sessions'])):
                segmentation = data['subjects'][i]['sessions'][j].get('segmentation')
                segmentation_presence.append(segmentation is not None)
                session = [
                    root_dir + data['subjects'][i]['sessions'][j]['image'],
                    root_dir + segmentation if segmentation is not None else None,
                    data['subjects'][i]['sessions'][j]['age'],
                ]
                subject.append(session)

            subject.sort(key=lambda session: session[2])
            if len(subject) >= 2:
                self.data_train.append(subject)

        with open(self.json_path_val, 'r') as f:
            # Parsing the JSON file into a Python dictionary
            data = json.load(f)
        for i in range(len(data['subjects'])):
            subject = []
            for j in range(len(data['subjects'][i]['sessions'])):
                segmentation = data['subjects'][i]['sessions'][j].get('segmentation')
                segmentation_presence.append(segmentation is not None)
                session = [
                    root_dir + data['subjects'][i]['sessions'][j]['image'],
                    root_dir + segmentation if segmentation is not None else None,
                    data['subjects'][i]['sessions'][j]['age'],
                ]
               
                subject.append(session)

            subject.sort(key=lambda session: session[2])
            if len(subject) >= 2:
                self.data_val.append(subject)

        if not self.data_train:
            raise ValueError("the training manifest contains no sequence with at least 2 sessions")
        if not self.data_val:
            raise ValueError("the validation manifest contains no sequence with at least 2 sessions")
        if any(segmentation_presence) and not all(segmentation_presence):
            raise ValueError(
                "segmentation availability must be dataset-wide: either every "
                "session has a segmentation or none of them does"
            )
        self.has_segmentation = all(segmentation_presence)

    def _loader_kwargs(self, *, shuffle: bool, pin_memory: bool) -> dict:
        """Build DataLoader options shared by arbitrary numbers of sequences."""
        options = {
            "batch_size": 1,
            "num_workers": self.num_workers,
            "shuffle": shuffle,
            "pin_memory": pin_memory,
            "persistent_workers": self.num_workers > 0,
            "drop_last": False,
        }
        if self.num_workers > 0:
            options["prefetch_factor"] = 1
        return options

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return a shuffled DataLoader over the training subjects."""
        dataset = SpatioTemporalDataset(
            self.data_train,
            self.transform,
            has_segmentation=self.has_segmentation,
        )
        return torch.utils.data.DataLoader(
            dataset=dataset, **self._loader_kwargs(shuffle=True, pin_memory=True)
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return an ordered DataLoader over the validation subjects."""
        dataset = SpatioTemporalDatasetValidation(
            self.data_val, self.transform, self.transform_seg,
            has_segmentation=self.has_segmentation,
        )
        return torch.utils.data.DataLoader(
            dataset=dataset, **self._loader_kwargs(shuffle=False, pin_memory=False)
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Return an ordered DataLoader over the validation subjects for testing."""
        dataset = SpatioTemporalDatasetValidation(
            self.data_val, self.transform, self.transform_seg,
            has_segmentation=self.has_segmentation,
        )
        return torch.utils.data.DataLoader(
            dataset=dataset, **self._loader_kwargs(shuffle=False, pin_memory=False)
        )
