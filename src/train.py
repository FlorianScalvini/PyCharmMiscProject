"""
Training entry-point for the longitudinal brain MRI registration pipeline.

Parses command-line arguments, builds a timestamped output directory, wires
together the data module, the :class:`~pl_module.RegistrationLongitudinal`
Lightning module, and a :class:`~pytorch_lightning.Trainer`, then launches
training — optionally resuming from an existing checkpoint.

Loss weights (``lambda_*``), learning rate, precision, and all scheduling
parameters are fully configurable via CLI flags.

Usage::

    python train.py --dataset data/macaque.yaml --max_epochs 5000

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
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# --- Local ---
from pl_module import RegistrationLongitudinal
from dataloader.dataloader import SpatioTemporalSequenceDatamoduleJSON



def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns
    -------
    Namespace
        Parsed arguments with the following attributes:

        Paths
        ~~~~~
        dataset : str
            Path to the YAML dataset configuration file.
        checkpoint : str or None
            Path to a checkpoint to resume training from.

        Data
        ~~~~
        batch_size : int
            Training batch size.
        num_workers : int
            Number of DataLoader worker processes.
        size : int
            Spatial size (isotropic) of the model input volume.

        Training
        ~~~~~~~~
        max_epochs : int
            Maximum number of training epochs.
        learning_rate : float
            Optimizer learning rate.
        lambda_seg : float
            Weight for the segmentation loss term.
        lambda_sim : float
            Weight for the image-similarity loss term.
        lambda_reg : float
            Weight for the regularisation loss term.
        lambda_jac : float
            Weight for the Jacobian-determinant loss term.
        precision : {16, 32}
            Floating-point precision used during training.
        num_sanity_val_steps : int
            Number of sanity validation steps before training starts.
        val_check_interval : int
            Run validation every N training iterations across epochs.
        checkpoint_every_n_steps : int
            Save a checkpoint every N training steps.
    """
    parser = argparse.ArgumentParser(
        description="Train the longitudinal brain MRI registration model."
    )

    # --- Paths ---
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/florian/PyCharmMiscProject/data/adni.yaml",
        help="Path to the dataset configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from (optional).",
    )

    # --- Data ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader worker processes.",
    )

    # --- Training ---
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5000,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--lambda_seg",
        type=float,
        default=0.0,
        help="Weight for the segmentation loss term.",
    )

    parser.add_argument(
        "--lambda_sim",
        type=float,
        default=1.0,
        help="Weight for the image-similarity loss term.",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.5,
        help="Weight for the regularisation loss term.",
    )
    parser.add_argument(
        "--lambda_jac",
        type=float,
        default=1,
        help="Weight for the Jacobian-determinant loss term.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32],
        help="Floating-point precision used during training.",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=0,
        help="Number of sanity validation steps before training starts.",
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=100,
        help="Run validation every N training iterations, across epochs.",
    )
    parser.add_argument(
        "--checkpoint_every_n_steps",
        type=int,
        default=100,
        help="Save a checkpoint every N training steps.",
    )
    parser.add_argument(
        "--force_last_target",
        action="store_true",
        help="Always use the last session as the training target.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs used per node.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of SLURM nodes used for distributed training.",
    )

    return parser.parse_args()


def main(args: Namespace) -> None:
    """Build all components and launch training.

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
    save_dir: str = os.path.join("./", "results", config["name"], "train", dir_name)
    if os.path.exists(save_dir):
        # create versioned directory if the base directory already exists
        version: int = 1
        while os.path.exists(f"{save_dir}_v{version}"):
            version += 1
        save_dir = f"{save_dir}_v{version}"
    os.makedirs(save_dir, exist_ok=True)

    # --- Logger ---
    tensorboard_logger: TensorBoardLogger = TensorBoardLogger(
        save_dir=save_dir,
        name="tensorboard",
        version=0,
        default_hp_metric=False,
    )

    # --- Data module ---
    datamodule: SpatioTemporalSequenceDatamoduleJSON = SpatioTemporalSequenceDatamoduleJSON(
        root_dir=config["root_dir"],
        json_path=config["train_json"],
        json_path_val=config["val_json"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        t0=config["t0"],
        tn=config["tn"],
    )

    # --- Model ---
    training_module: RegistrationLongitudinal = RegistrationLongitudinal(
        learning_rate=args.learning_rate,
        save_dir=save_dir,
        lambda_seg=args.lambda_seg,
        lambda_reg=args.lambda_reg,
        lambda_sim=args.lambda_sim,
        lambda_jac=args.lambda_jac,
        shape=config["rsize"],
        step_time=0.1,
        force_last_target=(
            args.force_last_target or config.get("force_last_target", False)
        ),
    )

    # --- Trainer ---
    trainer: pl.Trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="ddp" if args.devices * args.num_nodes > 1 else "auto",
        num_sanity_val_steps=args.num_sanity_val_steps,
        logger=tensorboard_logger,
        callbacks=[
            ModelCheckpoint(
                every_n_train_steps=args.checkpoint_every_n_steps,
                dirpath=save_dir,
                save_last=True,
            ),
            TQDMProgressBar(refresh_rate=1),
        ],
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=True,
    )
    trainer.fit(model=training_module, datamodule=datamodule, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main(args=parse_args())
