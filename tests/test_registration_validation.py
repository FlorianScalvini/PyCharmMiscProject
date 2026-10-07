"""Regression checks for validation, ODE regularization and deformation units."""

from types import SimpleNamespace

import monai
import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from model.neural_ode import ODEFunction
from pl_module import RegistrationLongitudinal
from utils.losses import NonDetJacobianPenalty
from utils.registration import generate_grid3d_tensor, phi_to_displacement_voxel


class ValidationHarness(RegistrationLongitudinal):
    """Exercise validation hooks without loading the registration network."""

    def __init__(self, save_dir):
        pl.LightningModule.__init__(self)
        self.model = nn.Linear(1, 1, bias=False)
        self.save_dir = str(save_dir)
        self.max_dice_score = 0
        self.seg_metrics = monai.metrics.DiceMetric()
        self._trainer = SimpleNamespace(sanity_checking=False, global_step=0)
        self.logged = {}
        self.scalars = {}
        self.on_validation_epoch_start()

    @property
    def logger(self):
        return SimpleNamespace(experiment=SimpleNamespace(add_scalar=self.record_scalar))

    def record_scalar(self, name, value, global_step):
        self.scalars[name] = value

    def log(self, name, value, **kwargs):
        self.logged[name] = value


def test_validation_aggregates_metrics_and_only_saves_improved_dice(tmp_path):
    module = ValidationHarness(tmp_path)
    module.table_result_data = [["0_1", 0.5, 2], ["1_1", 0.9, 6]]
    with torch.no_grad():
        module.model.weight.fill_(1)
    module.on_validation_epoch_end()
    assert module.logged["val/dice"] == pytest.approx(0.7)
    assert module.logged["val/jac_neg_count"] == pytest.approx(4)
    assert module.scalars["val/samples/0_1/dice"] == 0.5
    assert module.scalars["val/samples/1_1/jac_neg_count"] == 6
    assert module.table_result_data == []
    assert module.seg_metrics.get_buffer() is None
    checkpoint = tmp_path / "best_registration.pt"
    assert torch.load(checkpoint, weights_only=True)["weight"].item() == 1

    for score, weight, expected_weight in [(0.6, 2, 1), (0.8, 3, 3)]:
        with torch.no_grad():
            module.model.weight.fill_(weight)
        module.table_result_data = [["0_1", score, 0]]
        module.on_validation_epoch_end()
        assert torch.load(checkpoint, weights_only=True)["weight"].item() == expected_weight
    assert module.max_dice_score == 0.8


@pytest.mark.parametrize("sanity_checking", [False, True])
def test_validation_empty_or_sanity_epoch_does_not_save(tmp_path, sanity_checking):
    module = ValidationHarness(tmp_path)
    module._trainer.sanity_checking = sanity_checking
    if sanity_checking:
        module.table_result_data = [["0_1", 0.9, 0]]
    module.on_validation_epoch_end()
    assert module.logged == {}
    assert module.table_result_data == []
    assert not (tmp_path / "best_registration.pt").exists()


class ZeroVelocity(nn.Module):
    def forward(self, t, source, image_t, target, age_init, age_final):
        return source.new_zeros(source.shape[0], 3, *source.shape[2:])


class ConstantVelocity(nn.Module):
    def forward(self, t, source, image_t, target, age_init, age_final):
        return source.new_full((source.shape[0], 3, *source.shape[2:]), 2.0)


class ZeroRegularization(nn.Module):
    def forward(self, velocity):
        return velocity.new_zeros(())


class CaptureJacobian(nn.Module):
    def __init__(self):
        super().__init__()
        self.displacement = None
        self.penalty = NonDetJacobianPenalty()

    def forward(self, displacement):
        self.displacement = displacement
        return self.penalty(displacement)


def test_jacobian_uses_the_warping_voxel_displacement():
    shape = (7, 9, 11)
    grid = generate_grid3d_tensor(shape).unsqueeze(0)
    flow = torch.zeros_like(grid)
    # Compression close to the barrier makes the scale mismatch observable.
    flow[:, 0] = -0.96 * torch.arange(shape[0]).view(-1, 1, 1)
    d, h, w = shape
    xyz_scale = grid.new_tensor([w - 1, h - 1, d - 1]).view(1, 3, 1, 1, 1)
    phi = grid + 2 * flow[:, [2, 1, 0]] / xyz_scale
    source = torch.zeros(1, 1, *shape)
    jacobian = CaptureJacobian()
    duration = torch.tensor(0.2)
    ode = ODEFunction(
        ZeroVelocity(),
        source,
        source,
        identity_grid=grid,
        loss_jac=jacobian,
        ageA=torch.tensor(0.3),
        ageB=torch.tensor(0.3) + duration,
    )
    _, _, integrated_derivative = ode(
        torch.tensor(0.5), (phi, torch.tensor(0.0), torch.tensor(0.0))
    )
    external_flow = phi_to_displacement_voxel(phi, grid)
    torch.testing.assert_close(external_flow, flow, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(jacobian.displacement, external_flow, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        integrated_derivative,
        NonDetJacobianPenalty()(external_flow) * duration,
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_ode_regularization_uses_supplied_loss_and_integration_direction(direction):
    shape = (5, 5, 5)
    grid = generate_grid3d_tensor(shape).unsqueeze(0)
    source = torch.zeros(1, 1, *shape)

    class MeanSquareVelocity(nn.Module):
        def forward(self, velocity):
            return velocity.square().mean()

    ode = ODEFunction(
        ConstantVelocity(), source, source,
        ageA=torch.tensor(0.2), ageB=torch.tensor(0.2 + direction * 0.4),
        identity_grid=grid, loss_jac=NonDetJacobianPenalty(),
        loss_v=MeanSquareVelocity(),
    )
    dphi, regularization_derivative, jacobian_derivative = ode(
        torch.tensor(0.2), (grid, torch.tensor(0.0), torch.tensor(0.0))
    )
    torch.testing.assert_close(dphi, torch.full_like(grid, 2.0 * direction * 0.4))
    torch.testing.assert_close(regularization_derivative, torch.tensor(0.4 * 4.0))
    assert jacobian_derivative >= 0


def test_optimizer_step_decays_lr_and_resumes_scheduler(monkeypatch):
    from test_registration_endpoints import TrainingHarness

    module = TrainingHarness()
    monkeypatch.setattr(torch, "randint", lambda low, high, *args: torch.tensor(low))
    images = torch.rand(1, 3, 1, 5, 6, 7)
    batch = (images, torch.empty(1, 0), torch.tensor([[0.2, 0.4, 0.6]]))
    for step in range(3):
        module.training_step(batch, step)
        assert module.optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.999 ** (step + 1))
        assert module.logged["train/learning_rate"] == pytest.approx(0.01 * 0.999 ** step)
    resumed = TrainingHarness()
    resumed.optimizer.load_state_dict(module.optimizer.state_dict())
    resumed.scheduler.load_state_dict(module.scheduler.state_dict())
    resumed.training_step(batch, 3)
    assert resumed.scheduler.last_epoch == 4
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.999 ** 4)


def test_explicit_step_time_is_honoured(tmp_path):
    module = RegistrationLongitudinal(save_dir=str(tmp_path), shape=[16, 16, 16], step_time=0.125)
    assert module.model.step_time == 0.125
