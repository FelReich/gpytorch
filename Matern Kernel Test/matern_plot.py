import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import gpytorch
import matplotlib.pyplot as plt
import torch


def main():
    torch.manual_seed(0)

    n = 100
    #nus = [0.5, 1.5, 2.5, 3.5, 4.5]
    nus = [0.5, 2.5, 4.5]

    x = torch.linspace(0, 10, n).unsqueeze(-1).double()

    fig, axes = plt.subplots(len(nus), 1, figsize=(7, 7), sharex=True)

    with torch.no_grad():
        for ax, nu in zip(axes, nus):
            kernel = gpytorch.kernels.MaternKernel(nu=nu).double()
            kernel.initialize(lengthscale=0.5)

            K = kernel(x).to_dense()
            K = K + 1e-10 * torch.eye(n, dtype=K.dtype)

            L = torch.linalg.cholesky(K)

            eps = torch.randn(n, dtype=torch.double)
            y = L @ eps

            ax.plot(x.squeeze(-1).numpy(), y.numpy(), marker="o")
            ax.set_title(f"Matérn kernel, nu={nu}")
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("x")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()