import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import torch
import gpytorch


def main():
    print(gpytorch.__file__)

    x = torch.linspace(0, 1, 6).unsqueeze(-1).double()

    kernel = gpytorch.kernels.MaternKernel(nu=3.5).double()
    kernel.initialize(lengthscale=0.7)

    lazy_covar = kernel(x)
    covar = lazy_covar.to_dense()

    print(lazy_covar)
    print(covar)
    print(torch.linalg.eigvalsh(covar))
    print("symmetric:", torch.allclose(covar, covar.T))
    print("min eigenvalue:", torch.linalg.eigvalsh(covar).min().item())


if __name__ == "__main__":
    main()
