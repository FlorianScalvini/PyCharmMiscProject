# --- Standard library ---
import os

# --- Third-party ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import pytorch_lightning as pl
from torchvision.utils import save_image

# --- Local ---
import utils.utils as utils
import utils.losses as losses
import utils.registration as registration
from model.neural_ode import LongitudinalODERegistration


class RegistrationLongitudinal(pl.LightningModule):
    """PyTorch Lightning module for longitudinal brain image registration using Neural ODEs.

    Integrates a deformation-field ODE model with a multi-term loss (similarity,
    segmentation, regularisation and Jacobian determinant penalty) and logs
    training / validation metrics to TensorBoard.
    """

    # ──────────────────────────────────────────────────────────────────────────
    #  Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        learning_rate: float = 0.01,
        save_dir: str = "",
        lambda_seg: float = 1,
        lambda_reg: float = 0.001,
        lambda_sim: float = 0.0,
        lambda_jac: float = 0.000001,
        shape: list[int] = [192, 224, 192],
        step_time: float = 0.1,
        force_last_target: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialise model, loss functions, metrics, and tracking variables."""
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.learning_rate = learning_rate
        # Initialize the registration and segmentation networks
        self.model = LongitudinalODERegistration(
            shape=shape,
            step_time=step_time,
        )

        # Hyperparameters
        self.lambda_reg = lambda_reg
        self.lambda_sim = lambda_sim
        self.lambda_seg = lambda_seg
        self.lambda_jac = lambda_jac
        self.force_last_target = force_last_target
        # Loss functions and metrics
        self.loss_sim = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=9) # type: ignore
        self.loss_reg = losses.Grad3d('l2')
        self.loss_seg = nn.MSELoss()

        # Logging and tracking best performance
        self.save_dir = save_dir
        self.min_intensity_loss = float("inf")
        self.validation_intensity_losses = []
        self.validation_mae_values = []
        self.validation_psnr_values = []
        self.validation_negative_jacobian_percentages = []

        os.makedirs(os.path.join(self.save_dir, "parcellations"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "flows"), exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    #  Forward pass
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        ages: torch.Tensor,
        target_age: torch.Tensor,
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the ODE registration and rescale deformation fields to voxel space."""
        shape = source.shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        all_phi, loss_reg, loss_jac = self.model(
            source,
            target,
            ages,
            target_age,
            grid,
        )
        all_phi = (all_phi + 1.) / 2. * scale_factor
        return all_phi, loss_reg, loss_jac

    # ──────────────────────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self) -> tuple[list, list]:
        """Return Adam optimiser with exponential LR decay."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tuple, batch_idx: int) -> None:
        """Compute total weighted loss, back-propagate, and log per-term metrics."""
        optimizer = self.optimizers()

        images, segs, ages, has_segs = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)

        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        has_segs = has_segs.squeeze(0).bool()

        loss_sim = torch.tensor(0.0, device=self.device)
        loss_seg = torch.tensor(0.0, device=self.device)
        initial_img = images[0:1].float()
        target_idx = (
            images.shape[0] - 1
            if self.force_last_target
            else torch.randint(1, images.shape[0], (1,)).item()
        )
        target_img = images[target_idx:target_idx + 1].float()
        initial_seg = None
        if self.lambda_seg > 0 and bool(has_segs[0]):
            initial_seg = F.one_hot(
                segs[:, 0].squeeze(0).cpu().long(), num_classes=-1
            ).permute(0, 4, 1, 2, 3)
        seg_steps = 0
        all_phi, loss_reg, loss_jac = self(
            initial_img,
            target_img,
            ages,
            ages[target_idx],
            grid,
        )
        
        grid_voxel = (grid + 1.) / 2. * scale_factor

        for idx in range(1, images.shape[0]):
            phi = all_phi[idx]
            df = phi - grid_voxel
            if self.lambda_sim > 0:
                warped = registration.warp(initial_img, df)
                loss_sim += self.loss_sim(warped, images[idx:idx + 1].float())
                del warped
            if initial_seg is not None and bool(has_segs[idx]):
                warped_seg = registration.warp(initial_seg.float().to(self.device), df)
                loss_seg += self.loss_seg(warped_seg, F.one_hot(segs[:, idx].squeeze(0).cpu().long(), num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).float().to(self.device))
                seg_steps += 1
                del warped_seg
            del phi, df

        num_steps = images.shape[0] - 1
        if seg_steps > 0:
            loss_seg = loss_seg / seg_steps
        loss_sim = loss_sim / num_steps
        integration_duration = torch.abs(ages[-1] - ages[0]).clamp_min(1e-8)
        loss_reg = loss_reg / integration_duration
        loss_jac = loss_jac / integration_duration
        loss = (
            self.lambda_sim * loss_sim
            + self.lambda_seg * loss_seg
            + self.lambda_reg * loss_reg
            + self.lambda_jac * loss_jac
        )
        optimizer.zero_grad() # type: ignore
        self.manual_backward(loss)
        optimizer.step() # type: ignore

        self.log_dict(
            {
                "Train/Loss/Similarity": (self.lambda_sim * loss_sim).detach(),
                "Train/Loss/Segmentation": (self.lambda_seg * loss_seg).detach(),
                "Train/Loss/Regularization": (self.lambda_reg * loss_reg).detach(),
                "Train/Loss/Jacobian": (self.lambda_jac * loss_jac).detach(),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            "Train/Loss/Total",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

        # ── critical: free the ODE trajectory ──
        del all_phi, grid_voxel, loss, loss_sim, loss_reg
        # ── always flush at end of step ──
        torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        """Flush GPU cache and save a checkpoint at the end of each training epoch."""
        scheduler = self.lr_schedulers()
        scheduler.step()  # type: ignore[union-attr]
        torch.cuda.empty_cache()  # ← add this
        if self.trainer.is_global_zero:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_dir, "last_registration.pt"),
            )

    # ──────────────────────────────────────────────────────────────────────────
    #  Validation
    # ──────────────────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        """Reset per-epoch validation accumulators before the validation loop."""
        self.validation_intensity_losses = []
        self.validation_mae_values = []
        self.validation_psnr_values = []
        self.validation_negative_jacobian_percentages = []


    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Compute metrics and visualise the first ten validation sequences."""
        images, _segs, ages, _has_segs = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images.shape[2:]
        initial_img = images[0:1].float()
        target_img = images[-1:].float()
        all_phi, _, _ = self(
            initial_img,
            target_img,
            ages,
            ages[-1],
            grid,
        )
        grid_voxel = (grid + 1.) / 2. * scale_factor

        for idx in range(1, images.shape[0]):
            df = all_phi[idx] - grid_voxel
            warped = registration.warp(initial_img, df)
            target = images[idx:idx + 1].float()
            if self.lambda_sim > 0:
                intensity_loss = self.loss_sim(warped, target)
                self.validation_intensity_losses.append(
                    float(intensity_loss.cpu())
                )
            mae = F.l1_loss(warped, target)
            mse = F.mse_loss(warped, target)
            psnr = 10.0 * torch.log10(
                warped.new_tensor(1.0) / mse.clamp_min(1e-10)
            )
            self.validation_mae_values.append(float(mae.cpu()))
            self.validation_psnr_values.append(float(psnr.cpu()))
            if idx == images.shape[0] - 1:
                jacobian = utils.compute_jacobian_determinant_3d(df)
                negative_percentage = 100.0 * (jacobian < 0).float().mean()
                self.validation_negative_jacobian_percentages.append(
                    float(negative_percentage.cpu())
                )

        if (
            batch_idx < 10
            and self.trainer.is_global_zero
            and self.logger is not None
            and hasattr(self.logger.experiment, "add_image")
        ):
            axial_index = shape[-1] // 2
            source_slice = utils.normalize_to_0_1(
                initial_img[0, 0, :, :, axial_index]
            )
            target_slice = utils.normalize_to_0_1(
                images[-1, 0, :, :, axial_index]
            )
            warped_slice = utils.normalize_to_0_1(
                warped[0, 0, :, :, axial_index]
            )
            signed_difference = (
                warped[0, 0, :, :, axial_index]
                - images[-1, 0, :, :, axial_index]
            )
            difference_scale = signed_difference.abs().amax().clamp_min(1e-8)
            signed_difference = signed_difference / difference_scale
            positive_difference = signed_difference.clamp_min(0.0)
            negative_difference = (-signed_difference).clamp_min(0.0)
            difference_rgb = torch.stack(
                [
                    1.0 - negative_difference,
                    1.0 - signed_difference.abs(),
                    1.0 - positive_difference,
                ],
                dim=0,
            )
            axial_comparison = torch.cat(
                [
                    source_slice.unsqueeze(0).expand(3, -1, -1),
                    target_slice.unsqueeze(0).expand(3, -1, -1),
                    warped_slice.unsqueeze(0).expand(3, -1, -1),
                    difference_rgb,
                ],
                dim=2,
            )
            self.logger.experiment.add_image(
                f"Validation/Axial/Sequence_{batch_idx:02d}_Source_Target_Warped_Difference",
                axial_comparison.detach().cpu(),
                global_step=self.global_step,
            )

    def on_validation_epoch_end(self) -> None:
        """Log aggregated metrics and grid images; save model if a new Dice best is reached."""
        if self.validation_intensity_losses:
            mean_intensity_loss = float(np.mean(self.validation_intensity_losses))
            self.log(
                "Validation/Intensity/LNCC",
                mean_intensity_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            if (
                self.trainer.is_global_zero
                and mean_intensity_loss < self.min_intensity_loss
            ):
                self.min_intensity_loss = mean_intensity_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, "best_registration.pt"),
                )

        if self.validation_mae_values:
            mean_mae = float(np.mean(self.validation_mae_values))
            mean_psnr = float(np.mean(self.validation_psnr_values))
            self.log(
                "Validation/Intensity/MAE",
                mean_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "Validation/Intensity/PSNR",
                mean_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            if self.validation_negative_jacobian_percentages:
                mean_negative_jacobian = float(
                    np.mean(self.validation_negative_jacobian_percentages)
                )
                self.log(
                    "Validation/Deformation/NegativeJacobianPercent",
                    mean_negative_jacobian,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

        # Reset
        self.validation_intensity_losses = []
        self.validation_mae_values = []
        self.validation_psnr_values = []
        self.validation_negative_jacobian_percentages = []

        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    #  Test: PNG exports only
    # ──────────────────────────────────────────────────────────────────────────

    def on_test_start(self) -> None:
        """Create the directory used for lightweight test visualisations."""
        os.makedirs(os.path.join(self.save_dir, "png"), exist_ok=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Save central target/registered slices as PNG files."""
        images, _segs, ages, _has_segs = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape, device=self.device).view(
            1, 3, 1, 1, 1
        )
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        initial_img = images[0:1].float()

        all_phi, _, _ = self(
            initial_img,
            images[-1:].float(),
            ages,
            ages[-1],
            grid,
        )
        grid_voxel = (grid + 1.0) / 2.0 * scale_factor
        slice_index = shape[-1] // 2

        for time_idx in range(1, images.shape[0]):
            df = all_phi[time_idx] - grid_voxel
            warped = registration.warp(initial_img, df)
            target_slice = utils.normalize_to_0_1(
                images[time_idx, 0, :, :, slice_index]
            ).detach().cpu()
            warped_slice = utils.normalize_to_0_1(
                warped[0, 0, :, :, slice_index]
            ).detach().cpu()
            comparison = torch.stack([target_slice, warped_slice]).unsqueeze(1)
            save_image(
                comparison,
                os.path.join(
                    self.save_dir,
                    "png",
                    f"subject_{batch_idx:04d}_time_{time_idx:03d}.png",
                ),
                nrow=2,
                padding=4,
                pad_value=1.0,
            )
