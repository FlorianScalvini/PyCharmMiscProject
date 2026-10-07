"""Spatial augmentation must preserve alignment across acquisition times."""

import json

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from dataloader.dataloader import SpatioTemporalSequenceDatamoduleJSON
from dataloader.dataset import SpatioTemporalDataset


def write_sequence(tmp_path, affine=None):
    shape = (16, 18, 20)
    labels = np.zeros(shape, dtype=np.int16)
    labels[3:9, 4:11, 5:13] = 1
    labels[9:12, 5:8, 7:10] = 2
    image = labels.astype(np.float32) / 2
    affine = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(image, affine), tmp_path / "image.nii.gz")
    nib.save(nib.Nifti1Image(labels, affine), tmp_path / "labels.nii.gz")
    sessions = [
        {"image": "/image.nii.gz", "segmentation": "/labels.nii.gz", "age": age}
        for age in (4, 2, 3)
    ]
    (tmp_path / "data.json").write_text(json.dumps({"subjects": [{"sessions": sessions}]}))
    data = [[str(tmp_path / "image.nii.gz"), str(tmp_path / "labels.nii.gz"), age]
            for age in (4, 2, 3)]
    return shape, image, labels, data


@pytest.mark.parametrize("enabled", [False, True])
def test_augmentation_is_shared_across_dates_and_only_used_for_training(tmp_path, enabled):
    shape, _, _, _ = write_sequence(tmp_path)
    module = SpatioTemporalSequenceDatamoduleJSON(
        root_dir=str(tmp_path), json_path="data.json", json_path_val="data.json",
        batch_size=1, num_workers=0, size=shape, crop=shape,
        use_augmentation=enabled,
    )
    with torch.random.fork_rng():
        torch.manual_seed(12)
        images, labels, ages = module.train_dataloader().dataset[0]
    val_images, val_labels, val_ages = module.val_dataloader().dataset[0]
    test_images, test_labels, _ = module.test_dataloader().dataset[0]

    # Identical acquisitions remain identical after one random spatial transform.
    for idx in range(1, 3):
        torch.testing.assert_close(images[0], images[idx])
        torch.testing.assert_close(labels[0], labels[idx])
    torch.testing.assert_close(ages, torch.tensor([2., 3., 4.]))
    torch.testing.assert_close(ages, val_ages)
    assert images.shape == labels.shape == (3, 1, *shape)
    assert set(labels.unique().tolist()) <= {0, 1, 2}
    torch.testing.assert_close(test_images, val_images)
    torch.testing.assert_close(test_labels, val_labels)
    if enabled:
        assert not torch.equal(images, val_images)
        assert not torch.equal(labels, val_labels)
    else:
        torch.testing.assert_close(images, val_images)
        torch.testing.assert_close(labels, val_labels)


def test_left_right_flip_uses_anatomical_axis_and_preserves_image_label_alignment(tmp_path):
    # Here the anatomical left-right direction is the second voxel axis.
    affine = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                       [1., 0., 0., 0.], [0., 0., 0., 1.]])
    _, image, label, data = write_sequence(tmp_path, affine)
    dataset = SpatioTemporalDataset(
        [data], augmentation=tio.RandomFlip(axes=("LR",), flip_probability=1),
    )
    images, labels, _ = dataset[0]
    for idx in range(3):
        torch.testing.assert_close(images[idx, 0], torch.from_numpy(image).flip(1))
        torch.testing.assert_close(labels[idx, 0], torch.from_numpy(label).flip(1))


@pytest.mark.parametrize("partial", [False, True])
def test_missing_labels_preserve_images_and_ages(tmp_path, partial):
    shape, _, _, _ = write_sequence(tmp_path)
    manifest = json.loads((tmp_path / "data.json").read_text())
    sessions = manifest["subjects"][0]["sessions"]
    for session in sessions[:1] if partial else sessions:
        session.pop("segmentation")
    (tmp_path / "data.json").write_text(json.dumps(manifest))
    module = SpatioTemporalSequenceDatamoduleJSON(
        root_dir=str(tmp_path), json_path="data.json", json_path_val="data.json",
        batch_size=1, num_workers=0, size=shape, crop=shape, use_augmentation=True,
    )
    for loader in (module.train_dataloader(), module.val_dataloader(), module.test_dataloader()):
        images, labels, ages = loader.dataset[0]
        assert images.shape == (3, 1, *shape)
        assert labels.numel() == 0
        torch.testing.assert_close(ages, torch.tensor([2., 3., 4.]))
