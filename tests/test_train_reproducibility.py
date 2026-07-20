"""Tests for DataLoader worker seeding and balanced-sampler wiring."""

import random
from unittest import mock

import numpy as np
import torch

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
