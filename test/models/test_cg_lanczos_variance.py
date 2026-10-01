#!/usr/bin/env python3

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch
from linear_operator.operators import LinearOperator

import gpytorch


class _ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class TestCGLanczosVariance(unittest.TestCase):
    def setUp(self):
        self.train_x = torch.linspace(0, 1, 40, dtype=torch.float64)
        self.test_x = torch.linspace(0.05, 0.95, 6, dtype=torch.float64)

    def make_model(self, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        model = _ExactGP(self.train_x, train_y, likelihood).double()
        model.eval()
        likelihood.eval()
        return model

    @contextmanager
    def prediction_settings(self):
        with (
            gpytorch.settings.max_cholesky_size(0),
            gpytorch.settings.max_cg_iterations(100),
            gpytorch.settings.max_root_decomposition_size(25),
            gpytorch.settings.eval_cg_tolerance(1e-4),
            gpytorch.settings.fast_pred_var(True),
            gpytorch.settings.use_cg_lanczos_variance(True),
        ):
            yield

    def test_grad_enabled_prediction_with_detached_caches(self):
        model = self.make_model(torch.sin(6 * self.train_x))
        test_x = self.test_x.clone().requires_grad_()

        with self.prediction_settings():
            prediction = model(test_x)
            (prediction.mean.sum() + prediction.variance.sum()).backward()

        self.assertTrue(torch.isfinite(test_x.grad).all())
        self.assertIsNotNone(model.prediction_strategy.cg_lanczos_cache[1])

    def test_empty_basis_falls_back_to_love(self):
        model = self.make_model(torch.zeros_like(self.train_x))

        with torch.no_grad(), self.prediction_settings():
            with self.assertWarnsRegex(RuntimeWarning, "falling back to the standard LOVE"):
                prediction = model(self.test_x)
                variance = prediction.variance

        self.assertIsNone(model.prediction_strategy.cg_lanczos_cache[1])
        self.assertTrue(torch.isfinite(prediction.mean).all())
        self.assertTrue(torch.isfinite(variance).all())

    def test_missing_observation_policies_use_standard_path(self):
        train_y = torch.sin(6 * self.train_x)
        train_y[0] = float("nan")

        for policy in ("mask", "fill"):
            with self.subTest(policy=policy):
                model = self.make_model(train_y)
                with (
                    torch.no_grad(),
                    self.prediction_settings(),
                    gpytorch.settings.observation_nan_policy(policy),
                    patch.object(
                        LinearOperator, "solve_with_cg_lanczos_basis", side_effect=AssertionError("CG called")
                    ),
                ):
                    prediction = model(self.test_x)
                    self.assertTrue(torch.isfinite(prediction.mean).all())
                    self.assertTrue(torch.isfinite(prediction.variance).all())

    def test_fast_pred_var_off_uses_standard_mean(self):
        model = self.make_model(torch.sin(6 * self.train_x))

        with (
            torch.no_grad(),
            self.prediction_settings(),
            gpytorch.settings.fast_pred_var(False),
            patch.object(LinearOperator, "solve_with_cg_lanczos_basis", side_effect=AssertionError("CG called")),
        ):
            prediction = model(self.test_x)
            self.assertTrue(torch.isfinite(prediction.mean).all())
            self.assertTrue(torch.isfinite(prediction.variance).all())

    def test_direct_solve_settings_use_standard_prediction(self):
        for setting in (
            gpytorch.settings.max_cholesky_size(40),
            gpytorch.settings.max_cholesky_size(41),
            gpytorch.settings.fast_computations(solves=False),
        ):
            with self.subTest(setting=setting):
                model = self.make_model(torch.sin(6 * self.train_x))
                with (
                    torch.no_grad(),
                    self.prediction_settings(),
                    setting,
                    patch.object(LinearOperator, "solve_with_cg_lanczos_basis", side_effect=AssertionError("CG called")),
                ):
                    prediction = model(self.test_x)
                    self.assertTrue(torch.isfinite(prediction.mean).all())
                    self.assertTrue(torch.isfinite(prediction.variance).all())

    def test_operator_with_custom_solve_uses_standard_prediction(self):
        model = self.make_model(torch.sin(6 * self.train_x))
        model.covar_module = gpytorch.kernels.LinearKernel().double()

        with (
            torch.no_grad(),
            self.prediction_settings(),
            patch.object(LinearOperator, "solve_with_cg_lanczos_basis", side_effect=AssertionError("CG called")),
        ):
            prediction = model(self.test_x)
            self.assertTrue(torch.isfinite(prediction.mean).all())
            self.assertTrue(torch.isfinite(prediction.variance).all())

        self.assertEqual(
            type(model.prediction_strategy.train_train_covar_and_labels_offset[0].evaluate_kernel()).__name__,
            "LowRankRootAddedDiagLinearOperator",
        )

    def test_batched_prediction_uses_cg_lanczos(self):
        train_x = torch.linspace(0, 1, 12, dtype=torch.float64)
        train_y = torch.stack((torch.sin(6 * train_x), torch.cos(4 * train_x)))
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        model = _ExactGP(train_x, train_y, likelihood).double()
        model.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])), batch_shape=torch.Size([2])
        ).double()
        model.eval()
        likelihood.eval()

        with (
            torch.no_grad(),
            self.prediction_settings(),
            gpytorch.settings.max_cg_iterations(20),
            gpytorch.settings.max_root_decomposition_size(10),
        ):
            prediction = model(torch.linspace(0.05, 0.95, 4, dtype=torch.float64))
            self.assertTrue(torch.isfinite(prediction.mean).all())
            self.assertTrue(torch.isfinite(prediction.variance).all())

        self.assertEqual(prediction.mean.shape, (2, 4))
        self.assertEqual(model.prediction_strategy.cg_lanczos_cache[1].shape[:2], (2, 12))

    def test_gradients_with_attached_caches_use_standard_path(self):
        model = self.make_model(torch.sin(6 * self.train_x))
        test_x = self.test_x.clone().requires_grad_()

        with (
            self.prediction_settings(),
            gpytorch.settings.detach_test_caches(False),
            patch.object(LinearOperator, "solve_with_cg_lanczos_basis", side_effect=AssertionError("CG called")),
        ):
            prediction = model(test_x)
            (prediction.mean.sum() + prediction.variance.sum()).backward()

        self.assertTrue(torch.isfinite(test_x.grad).all())
        self.assertTrue(torch.isfinite(model.covar_module.base_kernel.raw_lengthscale.grad).all())


if __name__ == "__main__":
    unittest.main()
