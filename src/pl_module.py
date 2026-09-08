# --- Standard library ---
import os
import json

# --- Third-party ---
import torch
import torch.distributed as dist
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
        learning_rate: float = 0.001,
        save_dir: str = "",
        lambda_seg: float = 1,
        lambda_reg: float = 0.001,
        lambda_sim: float = 0.0,
        lambda_jac: float = 200.0,
        shape: list[int] = [192, 224, 192],
        step_time: float = 0.1,
        gradient_clip_norm: float = 1.0,
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
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        self.gradient_clip_norm = gradient_clip_norm
        # Loss functions and metrics
        self.loss_sim = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=7) # type: ignore
        self.loss_reg = losses.Grad3d('l2')
        self.loss_seg = nn.MSELoss()

        # Logging and tracking best performance
        self.save_dir = save_dir
        self.min_intensity_loss = float("inf")
        self.validation_intensity_losses = []
        self.validation_mae_values = []
        self.validation_psnr_values = []
        self.validation_negative_jacobian_percentages = []
        self.skipped_nonfinite_batches = 0
        self.validation_dice_values = []

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
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the ODE registration and rescale deformation fields to voxel space."""
        all_phi, loss_reg, loss_jac = self.model(
            source,
            target,
            ages,
            grid,
        )
        all_phi = torch.stack([
            registration.normalized_map_to_voxel_map(phi)
            for phi in all_phi
        ])
        return all_phi, loss_reg, loss_jac

    # ──────────────────────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self) -> tuple[list, list]:
        """Return Adam optimiser with exponential LR decay."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tuple, batch_idx: int) -> None:
        """Compute total weighted loss, back-propagate, and log per-term metrics."""
        optimizer = self.optimizers()

        images, segs, ages, has_segs = batch
        shape = images[0].shape[2:]
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)

        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        has_segs = has_segs.squeeze(0).bool()

        loss_sim = torch.tensor(0.0, device=self.device)
        loss_seg = torch.tensor(0.0, device=self.device)
        initial_img = images[0:1].float()
        target_img = images[-1:].float()
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
            grid,
        )
        
        grid_voxel = registration.normalized_map_to_voxel_map(grid)

        for idx in range(1, images.shape[0]):
            phi = all_phi[idx]
            df = phi - grid_voxel
            if self.lambda_sim > 0:
                warped = registration.warp(initial_img, df)
                loss_sim += self.loss_sim(
                    warped, images[idx:idx + 1].float()
                )
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
        if self._any_rank_has_nonfinite(loss.detach()):
            self._skip_nonfinite_batch(optimizer, ages, "loss")
            return

        self.manual_backward(loss)
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
        )
        if self._any_rank_flag(not gradients_finite):
            self._skip_nonfinite_batch(optimizer, ages, "gradients")
            return

        self._log_gradient_diagnostics()
        parameter_norm = torch.linalg.vector_norm(
            torch.stack([
                torch.linalg.vector_norm(parameter.detach())
                for parameter in self.model.parameters()
            ])
        )
        gradient_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            getattr(self, "gradient_clip_norm", 1.0),
            error_if_nonfinite=True,
        )
        clipped_gradients = gradient_norm_before_clip > getattr(
            self, "gradient_clip_norm", 1.0
        )
        clipped_gradient_values = [
            parameter.grad.detach()
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        gradient_norm_after_clip = torch.linalg.vector_norm(
            torch.stack([
                torch.linalg.vector_norm(gradient)
                for gradient in clipped_gradient_values
            ])
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        update_ratio = (
            learning_rate * gradient_norm_after_clip / parameter_norm.clamp_min(1e-12)
        )
        self.log_dict(
            {
                "Optimization/LearningRate": learning_rate,
                "Optimization/GradientNormBeforeClip": gradient_norm_before_clip.detach(),
                "Optimization/GradientNormAfterClip": gradient_norm_after_clip.detach(),
                "Optimization/GradientClipped": clipped_gradients.float().detach(),
                "Optimization/ParameterNorm": parameter_norm,
                "Optimization/UpdateRatioProxy": update_ratio.detach(),
            },
            on_step=True, on_epoch=True, batch_size=1, sync_dist=True,
        )
        optimizer.step() # type: ignore
        self.log(
            "Optimization/SkippedNonFiniteBatch",
            0.0,
            on_step=True,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

        final_displacement = all_phi[-1].detach() - grid_voxel
        final_jacobian = utils.compute_jacobian_determinant_3d(final_displacement)
        # Quantiles on a regular sample avoid sorting every voxel of the 3-D field.
        jacobian_sample = final_jacobian.flatten()[::4096]
        jacobian_quantiles = torch.quantile(
            jacobian_sample.float(),
            final_jacobian.new_tensor([0.01, 0.05, 0.5], dtype=torch.float32),
        )
        jacobian_diagnostics = {
            "Train/Jacobian/Minimum": final_jacobian.amin(),
            "Train/Jacobian/Quantile01": jacobian_quantiles[0],
            "Train/Jacobian/Quantile05": jacobian_quantiles[1],
            "Train/Jacobian/Median": jacobian_quantiles[2],
            "Train/Jacobian/PercentNegative": 100.0 * (final_jacobian < 0).float().mean(),
            "Train/Jacobian/PercentBelow005": 100.0 * (final_jacobian < 0.05).float().mean(),
            "Train/Jacobian/PercentBelow01": 100.0 * (final_jacobian < 0.1).float().mean(),
            "Train/Deformation/MaximumDisplacement": torch.linalg.vector_norm(
                final_displacement, dim=1
            ).amax(),
            "Train/Loss/JacobianRaw": loss_jac.detach(),
        }
        ode_diagnostics = getattr(self.model, "last_ode_diagnostics", {})
        step_diagnostics = {
            "Train/Step/LossTotal": loss.detach(),
            "Train/Step/LossSimilarity": loss_sim.detach(),
            "Train/Step/LossRegularizationRaw": loss_reg.detach(),
            "Train/Step/LossJacobianRaw": loss_jac.detach(),
            "Train/Step/AgeInitialNormalized": ages[0].detach(),
            "Train/Step/AgeFinalNormalized": ages[-1].detach(),
            "Train/Step/DurationNormalized": integration_duration.detach(),
        }
        ode_log_names = {
            "DiffusionRawMean": "Train/Step/LossDiffusionRawMean",
            "VelocityAmplitudeRawMean": "Train/Step/LossVelocityAmplitudeRawMean",
            "VelocityMeanNorm": "Train/Velocity/MeanNorm",
            "VelocityMaxNorm": "Train/Velocity/MaxNorm",
            "VelocityRMS": "Train/Velocity/RMS",
            "TrajectoryMinimumJacobian": "Train/Trajectory/MinimumJacobian",
            "TrajectoryMaximumDisplacement": "Train/Trajectory/MaximumDisplacement",
            "TauAtMaximumVelocity": "Train/Trajectory/TauAtMaximumVelocity",
            "TauAtMinimumJacobian": "Train/Trajectory/TauAtMinimumJacobian",
        }
        step_diagnostics.update({
            log_name: ode_diagnostics[name]
            for name, log_name in ode_log_names.items()
            if name in ode_diagnostics
        })
       

        self.log_dict(
            {
                "Train/Loss/Similarity": (self.lambda_sim * loss_sim).detach(),
                "Train/Loss/Segmentation": (self.lambda_seg * loss_seg).detach(),
                "Train/Loss/Regularization": (self.lambda_reg * loss_reg).detach(),
                "Train/Loss/Jacobian": (self.lambda_jac * loss_jac).detach(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
            sync_dist=True,
        )
        self.log_dict(
            jacobian_diagnostics,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
            sync_dist=True,
        )
        self.log_dict(
            step_diagnostics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "Train/Loss/Total",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            sync_dist=True,
        )

        # ── critical: free the ODE trajectory ──
        del all_phi, grid_voxel, loss, loss_sim, loss_reg
        # ── always flush at end of step ──
        torch.cuda.empty_cache()

    def _any_rank_has_nonfinite(self, value: torch.Tensor) -> bool:
        """Return whether a scalar/tensor is non-finite on at least one rank."""
        return self._any_rank_flag(not bool(torch.isfinite(value).all()))

    def _any_rank_flag(self, local_flag: bool) -> bool:
        """Synchronize a failure flag so every DDP rank takes the same branch."""
        invalid = torch.tensor(local_flag, device=self.device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        return bool(invalid.item())

    def _skip_nonfinite_batch(self, optimizer, ages: torch.Tensor, stage: str) -> None:
        """Discard a non-finite update while keeping all DDP ranks synchronized."""
        optimizer.zero_grad(set_to_none=True)
        self.skipped_nonfinite_batches += 1
        self.log(
            "Optimization/SkippedNonFiniteBatch",
            1.0,
            on_step=True,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if self.trainer.is_global_zero:
            age_values = ", ".join(f"{age:.6f}" for age in ages.detach().cpu())
            self.print(
                f"Skipping non-finite batch at global_step={self.global_step} "
                f"stage={stage} normalized_ages=[{age_values}] "
                f"total_skipped={self.skipped_nonfinite_batches}"
            )

    def _log_gradient_diagnostics(self) -> None:
        """Log gradient magnitudes and periodically record the output head."""
        gradients = [
            parameter.grad.detach()
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return

        global_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients])
        )
        global_max = torch.stack(
            [gradient.abs().amax() for gradient in gradients]
        ).amax()
        self.log(
            "Optimization/GradientNorm", global_norm,
            on_step=True, on_epoch=True, prog_bar=False,
            batch_size=1, sync_dist=True,
        )
        self.log(
            "Optimization/GradientMaxAbs", global_max,
            on_step=True, on_epoch=True, prog_bar=False,
            batch_size=1, sync_dist=True,
        )

        velocity_net = getattr(self.model, "velocity_net", None)
        output_layer = velocity_net.reg_head[-1] if velocity_net is not None else None
        output_gradient = (
            output_layer.weight.grad
            if isinstance(output_layer, nn.Conv3d)
            else None
        )
        if output_gradient is None:
            return

        output_gradient = output_gradient.detach()
        self.log(
            "Optimization/OutputHeadGradientNorm",
            torch.linalg.vector_norm(output_gradient),
            on_step=True, on_epoch=True, prog_bar=False,
            batch_size=1, sync_dist=True,
        )

        trainer = getattr(self, "_trainer", None)
        logger = trainer.logger if trainer is not None else None
        experiment = logger.experiment if logger is not None else None
        if (
            trainer is not None
            and trainer.is_global_zero
            and self.global_step % 100 == 0
            and experiment is not None
            and hasattr(experiment, "add_histogram")
        ):
            experiment.add_histogram(
                "Optimization/OutputHeadGradientHistogram",
                output_gradient.cpu(),
                global_step=self.global_step,
            )

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
        self.validation_dice_values = []

    @staticmethod
    def _validation_dice(prediction: torch.Tensor, target: torch.Tensor):
        """MONAI macro Dice including background and false-positive classes.

        Remap sparse label IDs jointly to avoid allocating unused classes.
        Labels absent from both maps (except background) are not evaluated.
        """
        labels = torch.unique(torch.cat((
            prediction.long().flatten(), target.long().flatten(),
            target.new_zeros(1, dtype=torch.long),
        )))
        prediction_indices = torch.searchsorted(labels, prediction.long())
        target_indices = torch.searchsorted(labels, target.long())
        metric = monai.metrics.DiceMetric(
            include_background=True,
            reduction="mean",
            ignore_empty=False,
            num_classes=labels.numel(),
        )
        metric(y_pred=prediction_indices, y=target_indices)
        return metric.aggregate().squeeze()


    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Compute metrics and visualise the first ten validation sequences."""
        images, segs, ages, has_segs = batch
        has_segs = has_segs.squeeze(0).bool()
        shape = images[0].shape[2:]
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
            grid,
        )
        grid_voxel = registration.normalized_map_to_voxel_map(grid)
        visualize_sequence = True
        source_one_hot = None
        if bool(has_segs[0]):
            source_labels = segs[:, 0, 0].long().to(self.device)
            label_values, label_indices = torch.unique(
                source_labels, sorted=True, return_inverse=True
            )
            source_one_hot = F.one_hot(
                label_indices, num_classes=label_values.numel()
            ).movedim(-1, 1).float()
        axial_index = shape[-1] // 2
        target_slices = []
        warped_slices = []
        difference_slices = []
        if visualize_sequence:
            source_slice = utils.normalize_to_0_1(
                initial_img[0, 0, :, :, axial_index]
            )
            source_rgb = source_slice.unsqueeze(0).expand(3, -1, -1)
            target_slices.append(source_rgb)
            warped_slices.append(source_rgb)
            difference_slices.append(torch.ones_like(source_rgb))

        for idx in range(1, images.shape[0]):
            df = all_phi[idx] - grid_voxel
            warped = registration.warp(initial_img, df)
            target = images[idx:idx + 1].float()
            if source_one_hot is not None and bool(has_segs[idx]):
                warped_probabilities = registration.warp(
                    source_one_hot, df, mode="bilinear"
                )
                warped_seg = label_values[
                    warped_probabilities.argmax(dim=1, keepdim=True)
                ]
                del warped_probabilities
                dice = self._validation_dice(
                    warped_seg, segs[:, idx].to(self.device)
                )
                if dice is not None:
                    self.validation_dice_values.append(float(dice.cpu()))
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

            if visualize_sequence:
                target_slice = utils.normalize_to_0_1(
                    target[0, 0, :, :, axial_index]
                )
                warped_slice = utils.normalize_to_0_1(
                    warped[0, 0, :, :, axial_index]
                )
                signed_difference = (
                    warped[0, 0, :, :, axial_index]
                    - target[0, 0, :, :, axial_index]
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
                target_slices.append(target_slice.unsqueeze(0).expand(3, -1, -1))
                warped_slices.append(warped_slice.unsqueeze(0).expand(3, -1, -1))
                difference_slices.append(difference_rgb)

        if visualize_sequence:
            axial_comparison = torch.cat(
                [
                    torch.cat(target_slices, dim=2),
                    torch.cat(warped_slices, dim=2),
                    torch.cat(difference_slices, dim=2),
                ],
                dim=1,
            )
            self.logger.experiment.add_image(
                f"Validation/Axial/Sequence_{batch_idx:02d}_Targets_Warped_Differences",
                axial_comparison.detach().cpu(),
                global_step=self.global_step,
            )

    def on_validation_epoch_end(self) -> None:
        """Log global validation means and select the checkpoint using global LNCC."""
        # Sequences have different numbers of comparisons. Reduce sums and
        # counts, not per-rank means, and include ranks with no observations.
        metric_values = (
            self.validation_intensity_losses,
            self.validation_mae_values,
            self.validation_psnr_values,
            self.validation_negative_jacobian_percentages,
            self.validation_dice_values,
        )
        statistics = torch.tensor(
            [[sum(values), len(values)] for values in metric_values],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        global_means = [
            total / count if count > 0 else None
            for total, count in statistics.cpu().tolist()
        ]
        mean_intensity_loss, mean_mae, mean_psnr, mean_negative_jacobian, mean_dice = global_means

        if mean_dice is not None:
            self.log(
                "Validation/Segmentation/Dice", mean_dice,
                on_step=False, on_epoch=True, prog_bar=True, sync_dist=False,
            )

        if mean_intensity_loss is not None:
            self.log(
                "Validation/Intensity/LNCC",
                mean_intensity_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,  # Already reduced across all ranks.
            )

            if mean_intensity_loss < self.min_intensity_loss:
                self.min_intensity_loss = mean_intensity_loss
                if self.trainer.is_global_zero:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_dir, "best_registration.pt"),
                    )

        if mean_mae is not None:
            self.log(
                "Validation/Intensity/MAE",
                mean_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )
        if mean_psnr is not None:
            self.log(
                "Validation/Intensity/PSNR",
                mean_psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )
        if mean_negative_jacobian is not None:
            self.log(
                "Validation/Deformation/NegativeJacobianPercent",
                mean_negative_jacobian,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )

        self.validation_intensity_losses = []
        self.validation_mae_values = []
        self.validation_psnr_values = []
        self.validation_negative_jacobian_percentages = []

        torch.cuda.empty_cache()
        self.validation_dice_values = []

    # ──────────────────────────────────────────────────────────────────────────
    #  Test: PNG exports only
    # ──────────────────────────────────────────────────────────────────────────

    def on_test_start(self) -> None:
        """Create output directories and reset test-set accumulators."""
        os.makedirs(os.path.join(self.save_dir, "png"), exist_ok=True)
        self.test_results = []

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Register every visit and save per-visit metrics and PNG comparisons."""
        images, _segs, ages, _has_segs = batch
        shape = images[0].shape[2:]
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        initial_img = images[0:1].float()

        all_phi, _, _ = self(
            initial_img,
            images[-1:].float(),
            ages,
            grid,
        )
        grid_voxel = registration.normalized_map_to_voxel_map(grid)
        slice_index = shape[-1] // 2
        sequence_paths = None
        test_dataloaders = getattr(self.trainer, "test_dataloaders", None)
        if test_dataloaders:
            dataset = test_dataloaders[0].dataset
            if hasattr(dataset, "data") and batch_idx < len(dataset.data):
                sequence_paths = dataset.data[batch_idx]

        for time_idx in range(1, images.shape[0]):
            df = all_phi[time_idx] - grid_voxel
            warped = registration.warp(initial_img, df)
            target = images[time_idx:time_idx + 1].float()
            lncc = self.loss_sim(warped, target)
            mae = F.l1_loss(warped, target)
            mse = F.mse_loss(warped, target)
            psnr = 10.0 * torch.log10(
                warped.new_tensor(1.0) / mse.clamp_min(1e-10)
            )
            jacobian = utils.compute_jacobian_determinant_3d(df)
            displacement_max = torch.linalg.vector_norm(df, dim=1).amax()
            self.test_results.append(
                {
                    "sequence": batch_idx,
                    "visit": time_idx,
                    "source_image": sequence_paths[0][0] if sequence_paths else None,
                    "target_image": sequence_paths[time_idx][0] if sequence_paths else None,
                    "source_age_normalized": float(ages[0].cpu()),
                    "target_age_normalized": float(ages[time_idx].cpu()),
                    "lncc": float(lncc.cpu()),
                    "mae": float(mae.cpu()),
                    "psnr": float(psnr.cpu()),
                    "jacobian_minimum": float(jacobian.amin().cpu()),
                    "jacobian_percent_negative": float(
                        (100.0 * (jacobian < 0).float().mean()).cpu()
                    ),
                    "jacobian_percent_below_005": float(
                        (100.0 * (jacobian < 0.05).float().mean()).cpu()
                    ),
                    "jacobian_percent_below_01": float(
                        (100.0 * (jacobian < 0.1).float().mean()).cpu()
                    ),
                    "maximum_displacement": float(displacement_max.cpu()),
                }
            )
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

    def on_test_epoch_end(self) -> None:
        """Log test-set means and write reproducible per-visit results."""
        if not self.test_results:
            return

        mean_metrics = {
            key: sum(result[key] for result in self.test_results) / len(self.test_results)
            for key in (
                "lncc", "mae", "psnr", "jacobian_percent_negative",
                "jacobian_percent_below_005", "jacobian_percent_below_01",
                "maximum_displacement",
            )
        }
        minimum_jacobian = min(
            result["jacobian_minimum"] for result in self.test_results
        )
        self.log_dict(
            {
                "Test/Intensity/LNCC": mean_metrics["lncc"],
                "Test/Intensity/MAE": mean_metrics["mae"],
                "Test/Intensity/PSNR": mean_metrics["psnr"],
                "Test/Deformation/JacobianMinimum": minimum_jacobian,
                "Test/Deformation/NegativeJacobianPercent": mean_metrics["jacobian_percent_negative"],
                "Test/Deformation/JacobianPercentBelow005": mean_metrics["jacobian_percent_below_005"],
                "Test/Deformation/JacobianPercentBelow01": mean_metrics["jacobian_percent_below_01"],
                "Test/Deformation/MaximumDisplacement": mean_metrics["maximum_displacement"],
            },
            sync_dist=False,
        )
        if self.trainer.is_global_zero:
            output = {
                "summary": {
                    **mean_metrics,
                    "jacobian_minimum": minimum_jacobian,
                    "number_of_sequences": len({result["sequence"] for result in self.test_results}),
                    "number_of_registered_visits": len(self.test_results),
                },
                "visits": self.test_results,
            }
            with open(os.path.join(self.save_dir, "test_results.json"), "w") as file:
                json.dump(output, file, indent=2)
