import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import torch
import gpytorch


def main():
    print(gpytorch.__file__)

    torch.manual_seed(0)

    x = torch.randn(8, 2, dtype=torch.double)

    kernel_fast = gpytorch.kernels.MaternKernel(nu=3.5).double()
    kernel_ref = gpytorch.kernels.MaternKernel(nu=3.5).double()

    kernel_fast.initialize(lengthscale=0.9)
    kernel_ref.initialize(lengthscale=0.9)

    # Fast path: x does not require grad, so this should use MaternCovariance.apply.
    covar_fast = kernel_fast(x).to_dense()
    loss_fast = covar_fast.sum()
    loss_fast.backward()
    grad_fast = kernel_fast.raw_lengthscale.grad.clone() #gradient (manually) calculated via fast path

    # Reference path: x requires grad, so this should use ordinary PyTorch ops.
    x_ref = x.clone().requires_grad_(True)
    covar_ref = kernel_ref(x_ref).to_dense()
    loss_ref = covar_ref.sum()
    loss_ref.backward()
    grad_ref = kernel_ref.raw_lengthscale.grad.clone() #gradient calculated via PyTorch

    print("covar_fast grad_fn:", covar_fast.grad_fn)
    print("covar_ref grad_fn:", covar_ref.grad_fn)
    print("max covar diff:", (covar_fast - covar_ref.detach()).abs().max().item())
    print("fast raw_lengthscale grad:", grad_fast)
    print("ref raw_lengthscale grad:", grad_ref)
    print("max grad diff:", (grad_fast - grad_ref).abs().max().item())


if __name__ == "__main__":
    main()