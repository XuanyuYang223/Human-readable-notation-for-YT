"""Tests for reproducible seeding and device selection."""

from __future__ import annotations

import random
import unittest
from unittest.mock import patch

import torch

from yt_transformer.runtime import resolve_device, seed_everything


class RuntimeTests(unittest.TestCase):
    def test_explicit_cpu_device(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_auto_falls_back_to_cpu_without_accelerators(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))

    def test_explicit_cuda_is_rejected_when_unavailable(self) -> None:
        if torch.cuda.is_available():
            self.assertEqual(resolve_device("cuda"), torch.device("cuda"))
        else:
            with self.assertRaisesRegex(ValueError, "CUDA.*not available"):
                resolve_device("cuda")

    def test_seed_everything_repeats_python_and_torch_sequences(self) -> None:
        seed_everything(1729)
        first_python = [random.random() for _ in range(4)]
        first_torch = torch.rand(4)

        seed_everything(1729)
        second_python = [random.random() for _ in range(4)]
        second_torch = torch.rand(4)

        self.assertEqual(first_python, second_python)
        self.assertTrue(torch.equal(first_torch, second_torch))

    def test_rejects_empty_and_nonexecution_devices(self) -> None:
        for requested in ("", "meta"):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError):
                    resolve_device(requested)

    def test_rejects_out_of_range_cuda_index(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
        ):
            with self.assertRaisesRegex(ValueError, "index 2"):
                resolve_device("cuda:2")


if __name__ == "__main__":
    unittest.main()
