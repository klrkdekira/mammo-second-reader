"""Tests for DataLoader worker seeding, augmentation seeding and the sampler."""

import random
from unittest import mock

import numpy as np
import torch

from src.data.augment import train_augment
from src.training.sampler import balanced_sampler
from src.training.train import _seed_worker


def test_seed_worker_makes_numpy_and_random_deterministic():
    # torch.initial_seed() is what a real worker reads; pin it and confirm
    # _seed_worker drives numpy/random to a reproducible, torch-derived state.
    def draw():
        with mock.patch.object(torch, "initial_seed", return_value=12345):
            _seed_worker(0)
        return np.random.rand(3).tolist(), [random.random() for _ in range(3)]

    assert draw() == draw()


def test_seed_worker_differs_with_worker_seed():
    with mock.patch.object(torch, "initial_seed", return_value=1):
        _seed_worker(0)
        a = np.random.rand(3).tolist()
    with mock.patch.object(torch, "initial_seed", return_value=2):
        _seed_worker(0)
        b = np.random.rand(3).tolist()
    assert a != b


def test_balanced_sampler_is_reproducible_with_generator():
    labels = [0] * 8 + [1] * 2

    def draw():
        g = torch.Generator()
        g.manual_seed(42)
        return list(balanced_sampler(labels, generator=g))

    assert draw() == draw()


def test_balanced_sampler_upweights_minority():
    labels = [0] * 90 + [1] * 10
    g = torch.Generator()
    g.manual_seed(0)
    drawn = list(balanced_sampler(labels, generator=g))
    minority = sum(1 for i in drawn if labels[i] == 1)
    # With inverse-frequency weights the minority class is drawn far above its
    # 10% base rate; expect roughly balanced sampling.
    assert 0.35 < minority / len(drawn) < 0.65


def _asymmetric_draws(seed, n=8):
    """Sum the left half only: a horizontal flip must change this number."""
    image = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    augment = train_augment(32, level="light", seed=seed)
    return [float(augment(image=image)["image"][:, :16].sum()) for _ in range(n)]


def test_seeded_augmentation_is_reproducible():
    # Albumentations 2.x transforms hold their own RNG, so this cannot be
    # covered by set_global_seed; the Compose seed is the only control.
    assert _asymmetric_draws(7) == _asymmetric_draws(7)


def test_seeded_augmentation_differs_between_seeds():
    assert _asymmetric_draws(7) != _asymmetric_draws(8)


def test_unseeded_augmentation_is_not_reproducible():
    # Documents why train_augment must be given a seed: without one, two
    # identically-configured runs see different augmentation streams.
    draws = [_asymmetric_draws(None, n=16) for _ in range(4)]
    assert len({tuple(draw) for draw in draws}) > 1


def test_seed_worker_reseeds_an_albumentations_transform():
    class _FakeDataset:
        def __init__(self):
            self.transform = train_augment(32, level="light", seed=1)

    dataset = _FakeDataset()
    info = mock.Mock(dataset=dataset)
    with (
        mock.patch.object(torch, "initial_seed", return_value=999),
        mock.patch.object(torch.utils.data, "get_worker_info", return_value=info),
    ):
        _seed_worker(0)
    image = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    after = [
        float(dataset.transform(image=image)["image"][:, :16].sum()) for _ in range(8)
    ]

    expected_pipeline = train_augment(32, level="light", seed=999)
    expected = [
        float(expected_pipeline(image=image)["image"][:, :16].sum()) for _ in range(8)
    ]
    assert after == expected
