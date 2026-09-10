"""Create one axial quality-control PNG per longitudinal sequence."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() and path.is_file():
        return path
    return root / value.lstrip("/")


def display_normalize(data: np.ndarray) -> np.ndarray:
    foreground = data[data != 0]
    if foreground.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lower, upper = np.percentile(foreground, (0.05, 99.5))
    if upper <= lower:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lower) / (upper - lower), 0, 1)


def signed_difference_rgb(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    difference = image - reference
    scale = max(float(np.max(np.abs(difference))), 1e-8)
    difference = difference / scale
    positive = np.clip(difference, 0, 1)
    negative = np.clip(-difference, 0, 1)
    return np.stack(
        [1 - negative, 1 - np.abs(difference), 1 - positive], axis=-1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=False, default=Path("/media/florian/SamsungDisk/Postdoc/Datasets/ADNI_crop"))
    parser.add_argument("--json", type=Path, help="Default: <root>/data.json")
    parser.add_argument("--output", type=Path, help="Default: ./qc_axial")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.json.resolve() if args.json else root / "data.json"
    output_directory = args.output.resolve() if args.output else root / "qc_axial"
    output_directory.mkdir(parents=True, exist_ok=True)
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    subjects = manifest.get("subjects", [])
    for subject_index, subject in enumerate(subjects, start=1):
        sessions = subject.get("sessions", [])
        if not sessions:
            continue
        volumes = []
        shapes = []
        for session in sessions:
            path = resolve_path(root, session["image"])
            if not path.is_file():
                raise FileNotFoundError(path)
            volume = np.asanyarray(nib.load(str(path)).dataobj)
            if volume.ndim != 3:
                raise ValueError(f"expected a 3-D image, got {volume.shape}: {path}")
            volumes.append(volume)
            shapes.append(volume.shape)
        if len(set(shapes)) != 1:
            raise ValueError(
                f"sessions of {subject['subject_id']} have different shapes: {shapes}"
            )

        axial_index = shapes[0][2] // 2
        slices = [display_normalize(volume[:, :, axial_index]) for volume in volumes]
        columns = len(slices)
        figure, axes = plt.subplots(
            2,
            columns,
            figsize=(4 * columns, 8),
            squeeze=False,
            constrained_layout=True,
        )
        reference = slices[0]
        for column, (session, image_slice) in enumerate(zip(sessions, slices)):
            axes[0, column].imshow(
                image_slice.T, cmap="gray", origin="lower", vmin=0, vmax=1
            )
            session_id = session.get("session_id", f"session-{column}")
            axes[0, column].set_title(f"{session_id} | age={session['age']:.2f}")
            axes[0, column].axis("off")
            axes[1, column].imshow(
                signed_difference_rgb(reference, image_slice).transpose(1, 0, 2),
                origin="lower",
            )
            axes[1, column].set_title("difference vs first")
            axes[1, column].axis("off")
        figure.suptitle(
            f"{subject['subject_id']} | axial slice {axial_index}", fontsize=14
        )
        output_path = output_directory / f"{subject['subject_id']}.png"
        figure.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        print(f"[{subject_index}/{len(subjects)}] {output_path}", flush=True)


if __name__ == "__main__":
    main()
