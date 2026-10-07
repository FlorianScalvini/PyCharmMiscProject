"""Check sampled anchor conditioning and intermediate supervision."""

import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from model.neural_ode import LongitudinalODERegistration, ODEFunction
from pl_module import RegistrationLongitudinal
from utils.registration import generate_grid3d_tensor, warp_with_phi


class ConstantVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(0.02))
        self.evaluation_times = []
        self.endpoint_ages = []

    def forward(self, t, source, image_t, target, age_init, age_final):
        self.evaluation_times.append(float(t.detach()))
        self.endpoint_ages.append((float(age_init), float(age_final)))
        return self.rate.expand(source.shape[0], 3, *source.shape[2:])


class ContractingEulerianVelocity(nn.Module):
    def __init__(self, grid, rate=-0.1):
        super().__init__()
        self.register_buffer("field", rate * grid)
        self.rate = rate

    def forward(self, t, source, image_t, target, age_init, age_final):
        return self.field.expand(source.shape[0], -1, -1, -1, -1)


def test_ode_passes_warped_source_to_velocity_net():
    shape = (7, 9, 11)
    source = torch.linspace(0, 1, shape[-1]).expand(1, 1, *shape).clone()
    target = torch.zeros_like(source)
    grid = generate_grid3d_tensor(shape).unsqueeze(0)
    phi = grid.clone()
    phi[:, 0] += 0.2

    class CaptureImages(nn.Module):
        def forward(self, t, image_a, image_t, image_b, age_a, age_b):
            self.received = (image_a, image_t, image_b, image_t.requires_grad)
            return torch.zeros_like(phi)

    velocity = CaptureImages()
    ode = ODEFunction(
        velocity, source, target, torch.tensor(0.2), torch.tensor(0.6),
        identity_grid=grid, loss_jac=nn.Identity(), loss_v=nn.Identity(),
    )
    ode(torch.tensor(0.4), (phi, source.new_zeros(()), source.new_zeros(())))
    image_a, image_t, image_b, requires_grad = velocity.received
    torch.testing.assert_close(image_a, source)
    torch.testing.assert_close(image_t, warp_with_phi(source, phi))
    torch.testing.assert_close(image_b, target)
    assert not requires_grad


def test_ode_integrates_over_acquisition_ages():
    shape = (7, 9, 11)
    model = LongitudinalODERegistration(shape=list(shape), step_time=0.1)
    velocity = ConstantVelocity()
    model.velocity_net = velocity
    source = torch.zeros(1, 1, *shape)
    target = torch.ones_like(source)
    ages = torch.tensor([0.2, 0.27, 0.4, 0.6])
    grid = generate_grid3d_tensor(shape).unsqueeze(0)

    trajectory, _, _ = model(source, target, ages, ages[-1], grid)

    relative_times = ages - ages[0]
    expected = grid.unsqueeze(0) + (
        relative_times.view(-1, 1, 1, 1, 1, 1) * velocity.rate
    )
    torch.testing.assert_close(trajectory, expected, atol=2e-6, rtol=1e-5)
    assert min(velocity.evaluation_times) == pytest.approx(float(ages[0]))
    assert max(velocity.evaluation_times) >= float(ages[-1]) - 1e-6
    for age_init, age_final in velocity.endpoint_ages:
        assert age_init == pytest.approx(float(ages[0]))
        assert age_final == pytest.approx(float(ages[-1]))
    trajectory[-1].mean().backward()
    assert velocity.rate.grad is not None
    assert torch.isfinite(velocity.rate.grad)
    torch.testing.assert_close(velocity.rate.grad, ages[-1] - ages[0])


def test_ode_composes_eulerian_velocity_at_current_positions():
    shape = (7, 9, 11)
    model = LongitudinalODERegistration(shape=list(shape), step_time=0.05)
    grid = generate_grid3d_tensor(shape).unsqueeze(0)
    velocity = ContractingEulerianVelocity(grid)
    model.velocity_net = velocity
    source = torch.zeros(1, 1, *shape)
    ages = torch.tensor([0.2, 0.6])

    trajectory, _, _ = model(source, source, ages, ages[-1], grid)

    expected = grid * torch.exp(velocity.rate * (ages[-1] - ages[0]))
    torch.testing.assert_close(
        trajectory[-1], expected, atol=2e-6, rtol=2e-6
    )


class RecordingTrajectory(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.01))
        self.received = None

    def forward(self, source, target, ages, target_age, grid):
        self.received = (source.detach().clone(), target.detach().clone(), ages.clone(), target_age.clone())
        relative_times = (ages - ages[0]) / (ages[-1] - ages[0])
        trajectory = grid.unsqueeze(0) + (
            relative_times.view(-1, 1, 1, 1, 1, 1) * self.offset
        )
        return trajectory, self.offset.square(), self.offset * 0


class RecordingSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.targets = []

    def forward(self, prediction, target):
        self.targets.append(target.detach().clone())
        return (prediction - target).square().mean()


class TrainingHarness(RegistrationLongitudinal):
    def __init__(self):
        pl.LightningModule.__init__(self)
        self.model = RecordingTrajectory()
        self.loss_sim = RecordingSimilarity()
        self.lambda_sim = 1.0
        self.lambda_seg = 0.0
        self.lambda_reg = 0.5
        self.lambda_jac = 0.5
        self.gradient_clip_norm = 1.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        self.logged = {}
        self.clip_arguments = None

    def optimizers(self):
        return self.optimizer

    def lr_schedulers(self):
        return self.scheduler

    def manual_backward(self, loss):
        loss.backward()

    def clip_gradients(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_arguments = (gradient_clip_val, gradient_clip_algorithm)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)

    def log_dict(self, values, **kwargs):
        self.logged.update(values)

    def log(self, name, value, **kwargs):
        self.logged[name] = value


@pytest.mark.parametrize("has_labels", [False, True])
def test_training_samples_anchors_and_supervises_full_suffix(monkeypatch, has_labels):
    module = TrainingHarness()
    shape = (5, 6, 7)
    base = torch.linspace(0, 0.5, shape[0]).view(1, -1, 1, 1).expand(1, *shape)
    images = torch.stack([base + 0.1 * index for index in range(4)]).unsqueeze(0)
    ages = torch.tensor([[2.63, 2.72, 6.0, 11.06]])
    segs = torch.zeros_like(images, dtype=torch.long)

    if not has_labels:
        segs = torch.empty(1, 0)
    # Start at visit 1, anchor at visit 2, and also supervise visit 3.
    draws = iter([1, 1])
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(next(draws)))
    before = module.model.offset.detach().clone()
    module.training_step((images, segs, ages), 0)

    source, target, received_ages, target_age = module.model.received
    torch.testing.assert_close(source, images[0, 1:2])
    torch.testing.assert_close(target, images[0, 2:3])
    torch.testing.assert_close(received_ages, ages[0, 1:])
    torch.testing.assert_close(target_age, ages[0, 2])
    assert len(module.loss_sim.targets) == 2
    for index, supervised_target in enumerate(module.loss_sim.targets, start=2):
        torch.testing.assert_close(supervised_target, images[0, index:index + 1])
    assert torch.isfinite(module.logged["train/loss"])
    assert module.clip_arguments == (1.0, "norm")
    assert module.model.offset.detach() != before


@pytest.mark.parametrize("scalar_ages", [False, True])
@pytest.mark.parametrize("current_age", [0.65, 0.9])
@pytest.mark.parametrize("use_absolute_age", [False, True])
def test_velocity_network_encodes_relative_and_absolute_ages_and_backpropagates(
    scalar_ages, current_age, use_absolute_age
):
    shape = (16, 16, 32)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        network = LongitudinalODERegistration(
            shape=list(shape), use_absolute_age=use_absolute_age
        ).velocity_net
        encoded_inputs = []
        hook = network.time_embedding.register_forward_pre_hook(
            lambda module, args: encoded_inputs.append(args[0].detach().clone())
        )
        image_inputs = []
        image_hook = network.encoder.register_forward_pre_hook(
            lambda module, args: image_inputs.append(args[0].detach().clone())
        )
        source = torch.rand(2, 1, *shape)
        image_t = source * 0.5
        target = torch.roll(source, 1, dims=2)
        t = torch.tensor(current_age)
        age_init = torch.tensor(0.6) if scalar_ages else torch.tensor([0.6, 0.4])
        age_final = torch.tensor(0.7) if scalar_ages else torch.tensor([0.7, 0.8])

        velocity = network(t, source, image_t, target, age_init, age_final)
        hook.remove()
        image_hook.remove()

        torch.testing.assert_close(image_inputs[0], torch.cat([source, image_t, target], dim=1))
        alpha = ((t - age_init) / (age_final - age_init)).expand(2)
        expected_inputs = (
            [t.expand(2), alpha, age_init.expand(2), age_final.expand(2)]
            if use_absolute_age else [alpha, (age_final - age_init).expand(2)]
        )
        assert len(encoded_inputs) == 1
        torch.testing.assert_close(
            encoded_inputs[0].reshape(2, -1), torch.stack(expected_inputs, dim=1)
        )
        assert velocity.shape == (2, 3, *shape)
        assert torch.isfinite(velocity).all()
        (velocity - 0.01).square().mean().backward()
        gradient = network.reg_head[-1].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        time_gradient = network.time_mlp[0].weight.grad
        assert time_gradient is not None and torch.isfinite(time_gradient).all()
        # Each temporal encoding must participate in learning.
        for block in time_gradient.split(network.t_dim_enc, dim=1):
            assert block.abs().sum() > 0
    finally:
        torch.set_num_threads(previous_threads)
