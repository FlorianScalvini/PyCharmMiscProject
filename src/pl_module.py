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
import utils.losses as losses
import utils.visualize as visualize
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
        gradient_clip_norm: float = 1.0,
        shape: list[int] = [192, 224, 192],
        step_time: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        """Initialise model, loss functions, metrics, and tracking variables."""
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.learning_rate = learning_rate
        # Initialize the registration and segmentation networks
        self.model = LongitudinalODERegistration(shape=shape, step_time=step_time)

        # Hyperparameters
        self.lambda_reg = lambda_reg
        self.lambda_sim = lambda_sim
        self.lambda_seg = lambda_seg
        self.lambda_jac = lambda_jac
        self.gradient_clip_norm = gradient_clip_norm
        # Loss functions and metrics
        self.loss_sim = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21) # type: ignore
        self.loss_reg = losses.Grad3d('l2')
        self.loss_seg = nn.MSELoss()

        self.seg_metrics = monai.metrics.DiceMetric() # type: ignore

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
        """Return Adam optimiser with exponential LR decay."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tuple, batch_idx: int) -> None:
        """Compute total weighted loss, back-propagate, and log per-term metrics."""
        optimizer = self.optimizers()

        images, segs, ages = batch
        shape = images[0].shape[2:]
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)

        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)

        loss_sim = torch.tensor(0.0, device=self.device)
        loss_seg = torch.tensor(0.0, device=self.device)
        initial_img = images[0:1].float()
        target_idx = images.shape[0] - 1
        target_img = images[target_idx:target_idx + 1].float()
        initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
        all_phi, loss_reg, loss_jac = self(
            initial_img, target_img, ages, ages[target_idx], grid
        )


        for idx in range(1, images.shape[0]):
            phi = all_phi[idx]
            if self.lambda_sim > 0:
                warped = registration.warp_with_phi(initial_img, phi)
                loss_sim += self.loss_sim(warped, images[idx:idx + 1].float())
                del warped
            if self.lambda_seg > 0:
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
        optimizer.zero_grad() # type: ignore
        self.manual_backward(loss)
        if self.gradient_clip_norm > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.gradient_clip_norm,
                gradient_clip_algorithm="norm",
            )
        optimizer.step() # type: ignore

        # One subject sequence per step; keep only the total in the progress bar.
        self.log(
            "train/loss", loss.detach(),
            on_step=False, on_epoch=True, prog_bar=True, batch_size=1,
        )
        self.log_dict({
            "train/loss_sim": (self.lambda_sim * loss_sim).detach(),
            "train/loss_seg": (self.lambda_seg * loss_seg).detach(),
            "train/loss_reg": (self.lambda_reg * loss_reg).detach(),
            "train/loss_jac": (self.lambda_jac * loss_jac).detach(),
        }, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)

        # ── critical: free the ODE trajectory ──
        del all_phi, loss, loss_sim, loss_reg
        # ── always flush at end of step ──
        torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        """Flush GPU cache and save a checkpoint at the end of each training epoch."""
        torch.cuda.empty_cache()  # ← add this
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "last_registration.pt"))

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

        initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
        for idx in range(0, images.shape[0]):
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
            warped_seg = registration.warp_with_phi(initial_seg.to(self.device).float(), phi)
            warped_seg = torch.argmax(warped_seg, dim=1).detach()
            save_label = reverse_transform(tio.LabelMap(tensor=warped_seg.int().cpu()))
            save_label.affine = subject_original_affine
            save_label.save(os.path.join(self.save_dir, "parcellations", f"segmentation_sample{batch_idx}_time{idx}.nii.gz"))
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

            pred_label = F.one_hot(warped_seg.cpu().long(), num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3)

            all_registered.append(
                utils.normalize_to_0_1(warped.squeeze())[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1, 1)
            )
            all_targets.append(
                utils.normalize_to_0_1(images[idx].squeeze(0))[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1,
                                                                                                                  1)
            )
            all_segs.append(
                utils.normalize_to_0_1(warped_seg.squeeze())[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1, 1)
            )
            xy = registration.displacement2grid(df.cpu()).squeeze(0).detach()
            grid_img = visualize.plt_grid(xy[:, :, shape[-1] // 2, :].cpu())[0]
            to_tensor = transforms.ToTensor()
            grid_img = to_tensor(grid_img)  # (3, H, W)

            if idx != 0:
                self.seg_metrics(pred_label, F.one_hot(segs[:, idx].squeeze(0).cpu().long(),
                                                       num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).cpu())
                det_jac = utils.compute_jacobian_determinant_3d(df.cpu()).numpy()
                nb_jac_neg = int(np.sum(det_jac < 0))
                buffer = self.seg_metrics.get_buffer()
                dice = float(buffer[-1].mean().item())
                results = [str(batch_idx) + "_" + str(idx), dice, nb_jac_neg]
                self.table_result_data.append(results)

            del warped, warped_seg, phi, xy, pred_label
            torch.cuda.empty_cache()

        del all_phi, df
        torch.cuda.empty_cache()

        num_times = images.shape[0]
        combined = torch.stack(all_targets + all_registered + all_segs)
        grid_visualization = make_grid(combined, nrow=num_times, padding=5, pad_value=1.0)
        self.logger.experiment.add_image(  # type: ignore
            f"val/images/sequence_{batch_idx:03d}",
            grid_visualization,
            global_step=self.global_step,
        )
        del combined, grid_visualization

    def on_validation_epoch_end(self) -> None:
        """Log aggregated metrics and grid images; save model if a new Dice best is reached."""
        if self.trainer.sanity_checking or not self.table_result_data:
            self.table_result_data = []
            self.seg_metrics.reset()
            return

        step = self.global_step

        dice_vals = [row[1] for row in self.table_result_data]
        jac_vals = [row[2] for row in self.table_result_data]

        # Log per-sample scalars
        for row in self.table_result_data:
            sample_id, dice, nb_jac_neg = row
            self.logger.experiment.add_scalar(f"val/samples/{sample_id}/dice", dice, global_step=step) # type: ignore
            self.logger.experiment.add_scalar(f"val/samples/{sample_id}/jac_neg_count", nb_jac_neg, global_step=step) # type: ignore

        mean_dice = float(np.mean(dice_vals))
        # Lightning writes each aggregate once, on the same global-step axis.
        self.log("val/dice", mean_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/jac_neg_count", float(np.mean(jac_vals)),
            on_step=False, on_epoch=True, prog_bar=True,
        )

        # Reset
        self.table_result_data = []
        self.seg_metrics.reset()

        if self.max_dice_score < mean_dice:
            self.max_dice_score = mean_dice
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_registration.pt"))

        torch.cuda.empty_cache()
'''
    # ──────────────────────────────────────────────────────────────────────────
    #  Test
    # ──────────────────────────────────────────────────────────────────────────

    def on_test_start(self) -> None:
        """Create output directories for images, parcellations, and flow fields."""
        # Create mri, seg and flows directories
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "parcellations"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "flows"), exist_ok=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Register, warp, and save NIfTI outputs for every time-point of a test subject."""
        images, segs, ages, = batch
        shape = images[0].shape[2:]
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images.shape[2:]
        initial_img = images[0:1].float()
        target_img = images[-1:].float()
        dices_subjects = []
        with torch.no_grad():
            all_phi, _, _ = self(initial_img, target_img, ages, ages[-1], grid)
        all_phi = all_phi.detach()
        subject = self.trainer.test_dataloaders.dataset.get_subject(batch_idx) # type: ignore
        affine = subject.image.affine
        reverse_transform = tio.transforms.CropOrPad(subject.image.shape[1:])
        initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
        for idx in range(0, images.shape[0]):
            phi = all_phi[idx]
            df = registration.phi_to_displacement_voxel(phi)
            warped = registration.warp_with_phi(images[0:1].float(), phi)
            warped_seg = registration.warp_with_phi(initial_seg.to(self.device).float(), phi)
            warped_seg = torch.argmax(warped_seg, dim=1).detach()
            image = reverse_transform(tio.ScalarImage(tensor=warped.cpu().squeeze(0).float()))
            image.affine = affine
            image.save(os.path.join(self.save_dir, "images", f"subject_{batch_idx}_time_{idx:03d}.nii.gz"))

            parcellation = reverse_transform(tio.LabelMap(tensor=warped_seg.cpu().float()))
            parcellation.affine = affine
            parcellation.save(os.path.join(self.save_dir, "parcellations", f"subject_{batch_idx}_time_{idx:03d}_seg.nii.gz"))

            df_image = reverse_transform(tio.ScalarImage(tensor=df.cpu().squeeze(0).float()))
            df_image.affine = affine
            spacing = subject.image.spacing
            # Voxel to mm conversion: multiply by voxel spacing
            df_image.data = df_image.data * torch.tensor(spacing).view(1, 3, 1, 1, 1)
            df_image.save(os.path.join(self.save_dir, "flows", f"subject_{batch_idx}_time_{idx:03d}_flow.nii.gz"))

            if idx != 0:
                pred_label = F.one_hot(warped_seg.cpu().long(), num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3)
                gt = F.one_hot(segs[:, idx].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3).cpu()
                dices_subjects.append(np.mean(self.seg_metrics(pred_label, gt.cpu()).numpy()))
            del warped, warped_seg, phi
            torch.cuda.empty_cache()
        print(f"Subject {batch_idx} : mean dice {np.mean(dices_subjects)}")
        del all_phi, df
        torch.cuda.empty_cache()

    def on_test_epoch_end(self) -> None:
        """Print final evaluation metrics and save the last model checkpoint."""
        print("Test epoch ended. Computing evaluation metrics...")
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "saved_model.pt"))
        torch.cuda.empty_cache()
'''
