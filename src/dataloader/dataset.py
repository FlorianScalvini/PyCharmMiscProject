"""
PyTorch Dataset classes for spatio-temporal longitudinal brain MRI sequences.

Each dataset wraps a list of per-subject session lists, where every session
entry is a ``(image_path, seg_path, age)`` triplet.  At index time the
datasets load all sessions for a given subject, optionally apply spatial
transforms, sort by acquisition age, and return stacked tensors ready for
the registration model.

Two variants are provided:

* :class:`SpatioTemporalDataset` — training dataset.  Filters out subjects
  with fewer than two sessions (single time-points cannot be used for
  longitudinal registration), loads images and segmentations, and sorts
  sessions by age before stacking.
* :class:`SpatioTemporalDatasetValidation` — validation / test dataset.
  Exposes :meth:`get_subject` so the test loop can retrieve the original
  TorchIO subject (with affine) for NIfTI export.

Author : Florian Scalvini
"""

# --- Third-party ---
import torch
import torchio as tio
from torchio import transforms


def _merge_first_two_labels(labels: torch.Tensor) -> torch.Tensor:
    """Merge labels 0 and 1 while keeping the remaining labels contiguous."""
    return (labels - 1).clamp_min_(0)


# ──────────────────────────────────────────────────────────────────────────────
#  Training dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalDataset(torch.utils.data.Dataset):
    """Longitudinal MRI dataset for training.

    Loads image volumes, segmentation label maps, and acquisition ages for
    each subject. Subjects with fewer than two sessions are silently discarded
    because at least two time-points are required for longitudinal
    registration. Sessions are sorted by age before being stacked into tensors.

    Parameters
    ----------
    data : list
        Outer list — one entry per subject.  Inner list — one entry per
        session, each being a ``[image_path, seg_path, age]`` triplet.
        *seg_path* may be ``None`` if no segmentation is available.
    transform : transforms.Transform or None
        Spatial transform applied to each image volume independently.
    augmentation : bool or transforms.Transform or None
        True enables a shared anterior-posterior flip (probability 0.5),
        followed by either axial rotation (+/-10 degrees) or isotropic scaling
        (0.9-1.1), and independent blur/noise for each session. False or None
        disables augmentation. A custom transform is applied to the sequence.
    merge_labels_0_1 : bool
        Merge labels 0 and 1 and shift higher labels down by one.
    """

    def __init__(
        self,
        data: list,
        transform: transforms.Transform | None = None,
        augmentation: bool | transforms.Transform | None = False,
        merge_labels_0_1: bool = False,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.intensity_augmentation = None
        if augmentation is True:
            self.augmentation = tio.Compose([
                tio.OneOf([
                    # RandomAffine uses physical axes; Z is superior-inferior.
                    tio.RandomAffine(scales=0, degrees=10),
                    tio.RandomAffine(scales=(0.9, 1.1), degrees=0, isotropic=True),
                    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, isotropic=True),
                ]),
            ])
            self.intensity_augmentation = tio.Compose([
                tio.RandomBlur(std=(0, 1)),
                # Inputs are normally rescaled to [0, 1] by the data module.
                tio.RandomNoise(mean=0, std=(0, 0.05)),
            ])
        else:
            self.augmentation = None if augmentation is False else augmentation
        self.merge_labels_0_1 = merge_labels_0_1
        self.data: list = []
        for i in range(len(data)):
            if len(data[i]) >= 2:
                self.data.append(sorted(data[i], key=lambda session: session[2]))

    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.data)


    def __getitem__(
        self, idx: int
    ) -> tuple:
        """Return all sessions for subject *idx* sorted by age.

        Parameters
        ----------
        idx : int
            Subject index.

        Returns
        -------
        mri_stack_out : torch.Tensor
            Stacked MRI volumes of shape ``(T, 1, X, Y, Z)``.
        seg_stack_out : torch.Tensor
            Stacked segmentation label maps of shape ``(T, 1, X, Y, Z)``.
        time_stack_out : torch.Tensor
            Acquisition ages of shape ``(T,)``.
        """
        mri_stack = []
        time_stack = []
        seg_stack = []
        data = self.data[idx]
        has_all_labels = all(session[1] is not None for session in data)
        sequence_images = {}
        for i in range(len(data)):
            subject_images = {'image': tio.ScalarImage(data[i][0])}
            if has_all_labels:
                subject_images['label'] = tio.LabelMap(data[i][1])
            session = tio.Subject(subject_images)
            if self.transform is not None:
                session = self.transform(session)
            sequence_images[f'image_{i}'] = session.image
            if has_all_labels:
                sequence_images[f'label_{i}'] = session.label

        # One call shares spatial parameters across all images and label maps.
        sequence = tio.Subject(sequence_images)
        if self.augmentation is not None:
            sequence = self.augmentation(sequence)
        for i in range(len(data)):
            image = sequence[f'image_{i}']
            if self.intensity_augmentation is not None:
                # Separate calls draw fresh blur/noise parameters per time point.
                image = self.intensity_augmentation(tio.Subject(image=image)).image
            mri_stack.append(image.data)
            if has_all_labels:
                labels = sequence[f'label_{i}'].data
                if self.merge_labels_0_1:
                    labels = _merge_first_two_labels(labels)
                seg_stack.append(labels)
            time_stack.append(data[i][2])

        # ── 5. stack ──────────────────────────────────────────────────
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)
        seg_stack_out = (
            torch.stack(seg_stack, dim=0) if seg_stack else torch.empty(0)
        )
        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)

        return mri_stack_out, seg_stack_out, time_stack_out

# ──────────────────────────────────────────────────────────────────────────────
#  Validation / test dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalDatasetValidation(torch.utils.data.Dataset):
    """Longitudinal MRI dataset for validation and testing.

    Keeps all subjects regardless of session count and exposes
    :meth:`get_subject` so the test loop can retrieve the full TorchIO subject
    (with affine matrix) for NIfTI-format saving.

    Parameters
    ----------
    data : list
        Outer list — one entry per subject.  Inner list — one entry per
        session, each being a ``[image_path, seg_path, age]`` triplet.
    transform : transforms.Transform or None
        Spatial transform applied to each full subject at load time.
    transform_seg : transforms.Transform or None
        Spatial transform applied to segmentation maps (reserved for API
        consistency; not applied inside ``__getitem__``).
    reverse_transform : transforms.Transform or None
        Inverse spatial transform used to map predictions back to the
        original subject space (e.g. :class:`tio.CropOrPad`).
    merge_labels_0_1 : bool
        Merge labels 0 and 1 and shift higher labels down by one.
    """

    def __init__(
        self,
        data: list,
        transform: transforms.Transform | None = None,
        transform_seg: transforms.Transform | None = None,
        reverse_transform: transforms.Transform | None = None,
        merge_labels_0_1: bool = False,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.transform_seg = transform_seg
        self.merge_labels_0_1 = merge_labels_0_1
        self.data = [
            sorted(subject, key=lambda session: session[2])
            for subject in data if len(subject) >= 2
        ]
        self.reverse_transform = reverse_transform
    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.data)

    def get_reverse_transform(self) -> transforms.Transform | None:
        """Return the inverse spatial transform, or ``None`` if not set."""
        return self.reverse_transform

    def get_subject(self, idx_subject: int, idx_session: int) -> tio.Subject:
        """Return the raw TorchIO subject at *idx* without applying transforms.

        Parameters
        ----------
        idx_subject : int
            Subject index.
        idx_session : int
            Session index.

        Returns
        -------
        tio.Subject
            Subject loaded from disk with ``image`` and (optionally) ``label``
            fields, preserving the original affine for NIfTI export.
        """
        data = self.data[idx_subject]
        subject_images = {'image': tio.ScalarImage(data[idx_session][0])}
        if data[idx_session][1] is not None:
            subject_images['label'] = tio.LabelMap(data[idx_session][1])
        session = tio.Subject(subject_images)
        return session

    def __getitem__(
        self, idx: int
    ) -> tuple:
        """Return all sessions for subject *idx* as stacked tensors.

        Parameters
        ----------
        idx : int
            Subject index.

        Returns
        -------
        mri_stack_out : torch.Tensor
            Stacked MRI volumes of shape ``(T, 1, X, Y, Z)``.
        seg_stack_out : torch.Tensor
            Stacked segmentation label maps of shape ``(T, 1, X, Y, Z)``,
            or an empty tensor if no labels are available.
        time_stack_out : torch.Tensor
            Acquisition ages of shape ``(T,)``.
        """
        mri_stack = []
        seg_stack = []
        time_stack = []
        data = self.data[idx]
        has_all_labels = all(session[1] is not None for session in data)
        for i in range(len(data)):
            subject_images = {'image': tio.ScalarImage(data[i][0])}
            if has_all_labels:
                subject_images['label'] = tio.LabelMap(data[i][1])
            session = tio.Subject(subject_images)
            if self.transform is not None:
                session = self.transform(session)

            mri_stack.append(session.image.data)
            if has_all_labels:
                labels = session.label.data
                if self.merge_labels_0_1:
                    labels = _merge_first_two_labels(labels)
                seg_stack.append(labels)
            time_stack.append(data[i][2])
            del session
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)

        if len(seg_stack) > 0:
            seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        else:
            seg_stack_out = torch.empty(0)

        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)
        return mri_stack_out, seg_stack_out, time_stack_out
