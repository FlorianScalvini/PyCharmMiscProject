# --- Standard library ---
import os
import json
import random

# --- Third-party ---
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import pytorch_lightning as pl
import torchio as tio
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from pytorch_lightning.utilities.types import STEP_OUTPUT

# --- Local ---
import utils.utils as utils
import utils.visualize as visualize
import utils.registration as registration
from model.neural_ode import LongitudinalODERegistration
from utils.training_diagnostics import (
    component_gradient_metrics, deformation_metrics, tensor_norm,
)


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
        gradient_clip_norm: float = 1.0,
        shape: list[int] = [192, 224, 192],
        step_time: float | None = None,
        use_absolute_age: bool = True,
        random_target: bool = False,
        diagnostic_every_n_steps: int = 50,
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
            step_time=0.05 if step_time is None else step_time,
            use_absolute_age=use_absolute_age,
        )

        # Hyperparameters
        self.lambda_reg = lambda_reg
        self.lambda_sim = lambda_sim
        self.lambda_seg = lambda_seg
        self.lambda_jac = lambda_jac
        self.gradient_clip_norm = gradient_clip_norm
        self.random_target = random_target
        if diagnostic_every_n_steps < 0:
            raise ValueError("diagnostic_every_n_steps must be nonnegative")
        self.diagnostic_every_n_steps = diagnostic_every_n_steps
        # Loss functions and metrics
        self.loss_sim = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21) # type: ignore
        self.loss_seg = nn.MSELoss()

        self.seg_metrics = monai.metrics.DiceMetric(ignore_empty=True) # type: ignore

        # Logging and tracking best performance
        self.save_dir = save_dir
        self.max_dice_score = 0
        self.table_result_data = []

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
        """Return absolute normalized deformation maps from the ODE."""
        all_phi, loss_reg, loss_jac = self.model(
            source, target, ages, target_age, grid
        )
        return all_phi, loss_reg, loss_jac

    # ──────────────────────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self) -> tuple[list, list]:
        """Return Adam with exponential learning-rate decay."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        return [optimizer], [scheduler]

    def training_step(self, batch: tuple, _) -> None:
        """Compute total weighted loss, back-propagate, and log per-term metrics."""
        optimizer = self.optimizers()

        images, segs, ages = batch
        shape = images[0].shape[2:]
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)

        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)

        # Start from any observed time point except the last one. Slicing the
        # three sequences together keeps images, labels and ages aligned.
        start_idx = 0
        images = images[start_idx:]
        has_segmentation = segs.numel() > 0
        if has_segmentation:
            segs = segs[:, start_idx:]
        ages = ages[start_idx:]

        loss_sim = torch.tensor(0.0, device=self.device)
        loss_seg = torch.tensor(0.0, device=self.device)
        initial_img = images[0:1].float()
        # Optionally sample any later session as the target anchor while still
        # supervising the full trajectory, including extrapolated sessions.
        target_idx = (
            int(torch.randint(1, images.shape[0], ()).item())
            if self.random_target
            else images.shape[0] - 1
        )
        target_img = images[target_idx:target_idx + 1].float()
        initial_seg = None
        if has_segmentation and self.lambda_seg > 0:
            initial_seg = F.one_hot(
                segs[:, 0].squeeze(0).cpu().long(), num_classes=-1
            ).permute(0, 4, 1, 2, 3)
        all_phi, loss_reg, loss_jac = self(
            initial_img, target_img, ages, ages[target_idx], grid
        )


        for idx in range(1, images.shape[0]):
            phi = all_phi[idx]
            if self.lambda_sim > 0:
                warped = registration.warp_with_phi(initial_img, phi)
                loss_sim += self.loss_sim(warped, images[idx:idx + 1].float())
                del warped
            if initial_seg is not None:
                warped_seg = registration.warp_with_phi(initial_seg.float().to(self.device), phi)
                loss_seg += self.loss_seg(warped_seg[:, :], F.one_hot(segs[:, idx].squeeze(0).cpu().long(), num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).float().to(self.device))
                del warped_seg
            del phi

        num_steps = images.shape[0] - 1
        loss_seg = loss_seg / num_steps
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
        parameters = tuple(p for p in self.model.parameters() if p.requires_grad)
        diagnostics = {}
        interval = self.diagnostic_every_n_steps
        if interval > 0 and self.global_step % interval == 0:
            trainer = getattr(self, "_trainer", None)
            single_device = trainer is None or trainer.world_size == 1
            diagnostics["component_gradients_enabled"] = float(single_device)
            if single_device:
                diagnostics.update(component_gradient_metrics({
                    "seg": self.lambda_seg * loss_seg,
                    "sim": self.lambda_sim * loss_sim,
                    "reg": self.lambda_reg * loss_reg,
                    "jac": self.lambda_jac * loss_jac,
                }, parameters))
            diagnostics.update(deformation_metrics(all_phi.detach(), grid))
        optimizer.zero_grad() # type: ignore
        solver_stats = getattr(self.model, "solver_stats", None)
        nfe_before_backward = solver_stats["nfe"] if solver_stats is not None else 0
        self.manual_backward(loss)
        if solver_stats is not None:
            diagnostics.update({
                "ode_forward_nfe": solver_stats["forward_nfe"],
                "ode_backward_nfe": solver_stats["nfe"] - nfe_before_backward,
                "ode_diagnostic_nfe": nfe_before_backward - solver_stats["forward_nfe"],
            })
        # In manual AMP optimization, Lightning unscales at optimizer.step(),
        # after this clipping call. Clip in scaled units and report true norms.
        trainer = getattr(self, "_trainer", None)
        scaler = getattr(getattr(trainer, "precision_plugin", None), "scaler", None)
        scale = float(scaler.get_scale()) if scaler is not None else 1.0
        pre_clip = tensor_norm((p.grad for p in parameters), loss) / scale
        if self.gradient_clip_norm > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.gradient_clip_norm * scale,
                gradient_clip_algorithm="norm",
            )
        post_clip = tensor_norm((p.grad for p in parameters), loss) / scale
        diagnostics.update({
            "sequence_length": ages.numel(),
            "age_start": ages[0].detach(),
            "age_end": ages[-1].detach(),
            "age_target": ages[target_idx].detach(),
            "integration_duration": integration_duration.detach(),
            "grad_norm_pre_clip": pre_clip,
            "grad_norm_post_clip": post_clip,
            "grad_finite": torch.isfinite(pre_clip).float(),
            "grad_clip_active": (
                (pre_clip > self.gradient_clip_norm).float()
                if self.gradient_clip_norm > 0 else loss.new_zeros(())
            ),
        })
        used_lr = optimizer.param_groups[0]["lr"]
        before_update = [p.detach().clone() for p in parameters]
        optimizer.step() # type: ignore
        diagnostics["parameter_update_norm"] = tensor_norm(
            (p.detach() - before for p, before in zip(parameters, before_update)), loss
        )
        del before_update
        self.lr_schedulers().step()
        self.log_dict({
            f"train/diagnostics/{name}": value for name, value in diagnostics.items()
        }, on_step=True, on_epoch=False, prog_bar=False, batch_size=1)
        self.log("train/learning_rate", used_lr, on_step=True, on_epoch=False, batch_size=1)

        # One subject sequence per step; keep only the total in the progress bar.
        self.log(
            "train/loss", loss.detach(),
            on_step=True, on_epoch=False, prog_bar=True, batch_size=1,
        )
        self.log_dict({
            "train/loss_sim": (self.lambda_sim * loss_sim).detach(),
            "train/loss_seg": (self.lambda_seg * loss_seg).detach(),
            "train/loss_reg": (self.lambda_reg * loss_reg).detach(),
            "train/loss_jac": (self.lambda_jac * loss_jac).detach(),
        }, on_step=True, on_epoch=False, prog_bar=False, batch_size=1)

        # ── critical: free the ODE trajectory ──
        del all_phi, loss, loss_sim, loss_reg
        # ── always flush at end of step ──
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    #  Validation
    # ──────────────────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        """Reset per-epoch validation accumulators before the validation loop."""
        self.table_result_data = []
        self.seg_metrics.reset()


    @staticmethod
    def _validation_reverse_transform(
        model_shape: tuple[int, int, int],
        crop_shape: tuple[int, int, int],
        original_shape: tuple[int, int, int],
    ) -> tio.transforms.Compose:
        """Map a validation prediction from model space to its original grid."""
        inverse = []
        if tuple(model_shape) != tuple(crop_shape):
            inverse.append(tio.transforms.Resize(crop_shape))
        if tuple(crop_shape) != tuple(original_shape):
            inverse.append(tio.transforms.CropOrPad(original_shape))
        return tio.transforms.Compose(inverse)


    @staticmethod
    def _save_displacement_nifti(
        flow_ras_mm: torch.Tensor,
        affine: np.ndarray,
        output_path: str,
    ) -> None:
        """Save ``(3,I,J,K)`` RAS-mm vectors as a NIfTI displacement field.

        Slicer requires a five-dimensional ``(I,J,K,1,3)`` image with NIfTI
        intent code 1006 (DISPVECT).  TorchIO's generic multichannel writer
        uses intent 1007 (VECTOR), which Slicer warns is not a transform.
        """
        if flow_ras_mm.ndim != 4 or flow_ras_mm.shape[0] != 3:
            raise ValueError(
                f"flow must have shape (3,I,J,K), got {tuple(flow_ras_mm.shape)}"
            )
        data = flow_ras_mm.permute(1, 2, 3, 0).unsqueeze(3).numpy()
        image = nib.Nifti1Image(data.astype(np.float32, copy=False), affine)
        image.header.set_intent(1006, name="displacement")
        image.header.set_xyzt_units(xyz="mm")
        nib.save(image, output_path)


    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Register images, compute Dice / Jacobian metrics, and collect visualisations."""
        # Initialization images
        all_registered = []
        all_targets = []
        all_segs = []

        images, segs, ages = batch
        shape = images[0].shape[2:]
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images.shape[2:]
        initial_img = images[0:1].float()
        target_img = images[-1:].float()
        with torch.no_grad():
            all_phi, _, _ = self(initial_img, target_img, ages, ages[-1], grid)
        all_phi = all_phi.detach()
        all_registered = []
        all_targets = []
        all_segs = []
        psnr_values = []
        ncc_loss_values = []
        save_validation_outputs = getattr(self.trainer, "world_size", 1) == 1

        has_segmentation = segs.numel() > 0
        initial_seg = None
        if has_segmentation:
            initial_seg = F.one_hot(
                segs[:, 0].squeeze(0).cpu().long(), num_classes=-1
            ).permute(0, 4, 1, 2, 3)
        for idx in range(0, images.shape[0]):
            if save_validation_outputs:
                original_session = self.trainer.val_dataloaders.dataset.get_subject(  # type: ignore
                    batch_idx, idx
                )
                subject_original_affine = original_session.image.affine
                original_shape = tuple(original_session.image.spatial_shape)
                crop_shape = tuple(
                    self.trainer.val_dataloaders.dataset.transform.transforms[0].target_shape  # type: ignore
                )
                reverse_transform = self._validation_reverse_transform(
                    tuple(shape), crop_shape, original_shape
                )
                processed_session = self.trainer.val_dataloaders.dataset.transform(  # type: ignore
                    original_session
                )
                model_affine = processed_session.image.affine
            phi = all_phi[idx]
            df = registration.phi_to_displacement_voxel(phi)
            warped = registration.warp_with_phi(images[0:1].float(), phi)
            if idx != 0:
                mse = F.mse_loss(warped, images[idx:idx + 1].float())
                psnr_values.append(-10 * torch.log10(mse.clamp_min(1e-10)))
                ncc_loss_values.append(self.loss_sim(warped, images[idx:idx + 1].float()))
            if initial_seg is not None:
                warped_seg = registration.warp_with_phi(
                    initial_seg.to(self.device).float(), phi
                )
                warped_seg = torch.argmax(warped_seg, dim=1).detach()
                if save_validation_outputs:
                    save_label = reverse_transform(
                        tio.LabelMap(tensor=warped_seg.int().cpu())
                    )
                    save_label.affine = subject_original_affine
                    save_label.save(os.path.join(self.save_dir, "parcellations", f"segmentation_sample{batch_idx}_time{idx}.nii.gz"))
            if save_validation_outputs:
                save_img = reverse_transform(tio.ScalarImage(tensor=warped.squeeze(0).cpu()))
                save_img.affine = subject_original_affine
                save_img.save(os.path.join(self.save_dir, "images", f"image_sample{batch_idx}_time{idx}.nii.gz"))

                # Save the exact model-grid displacement displayed in TensorBoard.
                # MONAI channels are dI,dJ,dK; ITK-SNAP needs physical RAS vectors.
                flow_ijk = df.squeeze(0).cpu()
                model_linear = torch.as_tensor(
                    model_affine[:3, :3], dtype=flow_ijk.dtype
                )
                flow_ras_mm = torch.einsum("rc,cijk->rijk", model_linear, flow_ijk)
                self._save_displacement_nifti(
                    flow_ras_mm,
                    model_affine,
                    os.path.join(
                        self.save_dir,
                        "flows",
                        f"df_sample{batch_idx}_time{idx}.nii.gz",
                    ),
                )

            pred_label = None
            if initial_seg is not None:
                pred_label = F.one_hot(
                    warped_seg.cpu().long(), num_classes=initial_seg.shape[1]
                ).permute(0, 4, 1, 2, 3)

            all_registered.append(
                utils.normalize_to_0_1(warped.squeeze())[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1, 1)
            )
            all_targets.append(
                utils.normalize_to_0_1(images[idx].squeeze(0))[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1,
                                                                                                                  1)
            )
            if initial_seg is not None:
                all_segs.append(
                    utils.normalize_to_0_1(warped_seg.squeeze())[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1, 1)
                )
            xy = registration.displacement2grid(df.cpu()).squeeze(0).detach()
            grid_img = visualize.plt_grid(xy[:, :, shape[-1] // 2, :].cpu())[0]
            to_tensor = transforms.ToTensor()
            grid_img = to_tensor(grid_img)  # (3, H, W)

            if idx != 0 and pred_label is not None and initial_seg is not None:
                self.seg_metrics(pred_label, F.one_hot(segs[:, idx].squeeze(0).cpu().long(),
                                                       num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).cpu())
                det_jac = utils.compute_jacobian_determinant_3d(df.cpu()).numpy()
                nb_jac_neg = float(np.sum(det_jac <= 0))
                buffer = self.seg_metrics.get_buffer()
                dice = float(torch.nanmean(buffer[-1]).item())
                results = [str(batch_idx) + "_" + str(idx), dice, nb_jac_neg]
                self.table_result_data.append(results)

            del warped, phi, xy
            if initial_seg is not None:
                del warped_seg, pred_label
            torch.cuda.empty_cache()

        del all_phi, df
        torch.cuda.empty_cache()

        if psnr_values:
            self.log(
                "val/loss_ncc",
                torch.stack(ncc_loss_values).mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,
                sync_dist=True,
            )
            self.log(
                "val/psnr",
                torch.stack(psnr_values).mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=1,
                sync_dist=True,
            )

        num_times = images.shape[0]
        combined = torch.stack(all_targets + all_registered + all_segs)
        grid_visualization = make_grid(combined, nrow=num_times, padding=5, pad_value=1.0)
        if getattr(self.trainer, "is_global_zero", True):
            self.logger.experiment.add_image(  # type: ignore
                f"val/images/sequence_{batch_idx:03d}",
                grid_visualization,
                global_step=self.global_step,
            )
        del combined, grid_visualization

    def on_validation_epoch_end(self) -> None:
        """Log aggregated metrics and grid images; save model if a new Dice best is reached."""
        if self.trainer.sanity_checking:
            self.table_result_data = []
            self.seg_metrics.reset()
            return

        step = self.global_step

        dice_vals = [row[1] for row in self.table_result_data]
        jac_vals = [row[2] for row in self.table_result_data]

        metric_totals = torch.tensor(
            [sum(dice_vals), sum(jac_vals), len(dice_vals)],
            dtype=torch.float64,
            device=self.device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(metric_totals)
        if metric_totals[2].item() == 0:
            self.table_result_data = []
            self.seg_metrics.reset()
            return
        mean_dice = float((metric_totals[0] / metric_totals[2]).item())
        mean_jac = float((metric_totals[1] / metric_totals[2]).item())

        # Log per-sample scalars
        if getattr(self.trainer, "is_global_zero", True):
            for row in self.table_result_data:
                sample_id, dice, nb_jac_neg = row
                self.logger.experiment.add_scalar(f"val/samples/{sample_id}/dice", dice, global_step=step) # type: ignore
                self.logger.experiment.add_scalar(f"val/samples/{sample_id}/jac_neg_count", float(nb_jac_neg), global_step=step) # type: ignore

        # Lightning writes each aggregate once, on the same global-step axis.
        self.log("val/dice", mean_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/jac_neg_count", mean_jac,
            on_step=False, on_epoch=True, prog_bar=True,
        )

        # Reset
        self.table_result_data = []
        self.seg_metrics.reset()

        if self.max_dice_score < mean_dice:
            self.max_dice_score = mean_dice
            if getattr(self.trainer, "is_global_zero", True):
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_registration.pt"))

        torch.cuda.empty_cache()
