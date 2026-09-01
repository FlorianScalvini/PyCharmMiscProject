"""
Test entry-point for the longitudinal brain MRI registration pipeline.

Loads a trained :class:`RegistrationLongitudinal` model from a checkpoint,
runs inference on the test split defined by the dataset YAML, and then
evaluates the predictions by computing segmentation Dice scores, the number
of non-positive Jacobian determinants (deformation regularity), gyrification
index (GI), and optional VTK / image-slice exports.

Author : Florian Scalvini
"""

# --- Standard library ---
import gc
import os
import argparse
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict

# --- Third-party ---
import torch
import yaml
import pytorch_lightning as pl

# --- Local ---
from pl_module import RegistrationLongitudinal
from dataloader.dataloader import SpatioTemporalSequenceDatamoduleJSON




def main(args: Namespace) -> None:
    """Build all components and launch testing.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments returned by :func:`parse_args`.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")

    with open(args.dataset, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # --- Output directory ---
    dir_name: str = datetime.now().strftime("%y_%d_%H_%M")
    save_dir: str = os.path.join("./", "results", config["name"], "test", dir_name)
    if os.path.exists(save_dir):
        # create versioned directory if the base directory already exists
        version: int = 1
        while os.path.exists(f"{save_dir}_v{version}"):
            version += 1
        save_dir = f"{save_dir}_v{version}"
    os.makedirs(save_dir, exist_ok=True)

    # --- Data module ---
    datamodule: pl.LightningDataModule = SpatioTemporalSequenceDatamoduleJSON(
        root_dir=config["root_dir"],
        json_path=config["train_json"],
        json_path_val=config.get("test_json", config["val_json"]),
        batch_size=1,
        num_workers=1,
        size=config["rsize"],
        crop=config["csize"],
        t0=config["t0"],
        tn=config["tn"],
    )

    # --- Model ---
    training_module: RegistrationLongitudinal = RegistrationLongitudinal(
        save_dir=save_dir,
        shape=config["rsize"],
        step_time=0.1
    )

    # --- Trainer ---
    trainer: pl.Trainer = pl.Trainer(
        precision=32,
        enable_progress_bar=True,
    )

    training_module.model.load_state_dict(torch.load(args.weight_path, weights_only=True))
    
    # Generate lightweight PNG comparisons.
    trainer.test(model=training_module, datamodule=datamodule)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the longitudinal brain MRI registration model."
    )
    # --- Paths ---
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/florian/PyCharmMiscProject/data/dhcpatlas.yaml",
        help="Path to the dataset configuration file.",
    )
    
    parser.add_argument(
        "--weight_path",
        type=str,
        default="/home/florian/PyCharmMiscProject/results/dhcpatlas/train/26_07_15_55_v1/last_registration.pt",
        help="Path to the weight file for testing.",
    )
  
    args = parser.parse_args()
    main(args=args)
