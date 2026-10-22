"""
algorithms
==========

Perform data-driven connected components on M_2.
"""

import torch
import torch.nn.functional as F
from math import sqrt, ceil, pi
from skimage.measure import label
from functools import lru_cache
from typing import Literal
from collections.abc import Callable
from tqdm.auto import tqdm


# Connected Components

def connected_components(
    on_data: torch.Tensor,
    cost: torch.Tensor,
    δ: float,
    G_inv: tuple[float, float, float],
    threshold: float = 0.8,
    projective_line_bundle: bool = True,
) -> list[torch.Tensor]:
    """
    Compute the connected components of binarised data `on_data` on M_2.[1]

    Args:
        `on_data`: binary torch.Tensor of shape `[Or, H, W]` representing
          structure on which to perform connected components.
        `cost`: `[Or, H, W]` representing the cost function.
        `δ`: radius of ball , taking values greater than 0.
        `G_inv`: tuple[float, float, float] of constants of the inverse of the
          diagonal metric tensor with respect to left invariant basis used to
          compute the distance. By setting `G_inv[1] = 0.`, we get a
          sub-Riemannian distance.
      Optional:
        `threshold`: threshold for rebinarising data after performing dilations.
          Defaults to 0.8.
        `projective_line_bundle`: boolean to choose between connected components
          on M_2 (False) or the projective line bundle (True). In the latter
          case, we identify (x, θ) ~ (x, θ + π). Defaults to True.

    Returns:
        List[torch.Tensor] of components given as binary arrays.

    References:
        [1]: N.J. van den Berg, O. Mula, L. Vis, and R. Duits.
          "Connected Components on Lie Groups and Applications to Multi-
          Orientation Image Analysis." In: Journal of Mathematical
          Imaging and Vision (2026).
          DOI:10.1007/s10851-026-01287-9.
    """
    Or = on_data.shape[0]
    dθ = 2 * pi / (Or - 1)
    dt_opt = compute_timestep(cost, dθ, G_inv)
    n = ceil(δ / (2.0 * dt_opt))
    dt = δ / (2.0 * n)
    print("Number of time steps to dilate: n = ", n)

    on_data_grown = on_data.clone()
    for _ in tqdm(range(n)):
        on_data_grown = step_connected_components(on_data_grown, cost, G_inv, dt)
    on_data_grown = on_data_grown > threshold

    grouped, N_components = label_components(on_data_grown, on_data, projective_line_bundle=projective_line_bundle)
    print(f"Found {N_components} components")

    return grouped


def persistence(
    on_data: torch.Tensor,
    cost: torch.Tensor,
    δs: torch.Tensor,
    G_inv: tuple[float, float, float],
    projective_line_bundle: bool = True,
    n_max: float = 1e4,
    tol: float = 1e-3,
    initial_condition: float = 1e8,
) -> list[torch.Tensor]:
    """
    Compute connected components persistence diagram of binarised data `on_data` on M_2.[1]

    Args:
        `on_data`: binary torch.Tensor of shape `[Or, H, W]` representing
          structure on which to perform connected components.
        `cost`: `[Or, H, W]` representing the cost function.
        `δ`: radius of ball , taking values greater than 0.
        `G_inv`: tuple[float, float, float] of constants of the inverse of the
          diagonal metric tensor with respect to left invariant basis used to
          compute the distance. By setting `G_inv[1] = 0.`, we get a
          sub-Riemannian distance.
      Optional:
        `projective_line_bundle`: boolean to choose between connected components
          on M_2 (False) or the projective line bundle (True). In the latter
          case, we identify (x, θ) ~ (x, θ + π). Defaults to True.

    Returns:
        List[torch.Tensor] of components given as binary arrays.

    References:
        [1]: N.J. van den Berg, O. Mula, L. Vis, and R. Duits.
          "Connected Components on Lie Groups and Applications to Multi-
          Orientation Image Analysis." In: Journal of Mathematical
          Imaging and Vision (2026).
          DOI:10.1007/s10851-026-01287-9.
    """
    device = on_data.device
    Or = on_data.shape[0]
    dθ = 2 * pi / Or
    # Solving u_t = cost - |∇u| (rather than 1 - |∇u| / cost) gives a CFL
    # condition independent of the cost.
    dt = 1. / sqrt(G_inv[0] + G_inv[1] + G_inv[2] / dθ**2)

    u = torch.ones(cost.shape, device=device) * initial_condition
    # Fix u = 0 on the data, and let it evolve elsewhere.
    boundary_condition = (on_data <= 0.0).float()
    u = u * boundary_condition

    error = torch.inf
    is_converged = False
    with tqdm(range(int(n_max))) as pbar:
        for n in pbar:
            u, du_dt = step_distance(u, cost, G_inv, dt)
            u = u * boundary_condition
            error = check_convergence(du_dt, boundary_condition)
            if n % 50 == 0:
                pbar.set_postfix({"error": error.item()})
            if error < tol:
                is_converged = True
                print(f"Converged after {n} steps!")
                break
    if not is_converged:
        print(f"Hamiltonian did not converge to tolerance {tol}!")

    alive_components = []
    for δ in δs:
        on_data_grown = u <= δ
        _, N_components = label_components(on_data_grown, on_data, projective_line_bundle=projective_line_bundle)
        alive_components.append(N_components)

    return u, alive_components


# Helper Functions


def label_components(on_data_grown: torch.Tensor, on_data: torch.Tensor, projective_line_bundle=True):
    grouped = torch.tensor(
        label(on_data_grown.cpu()),
        device=on_data.device,
    ) * (on_data > 0.0)

    if projective_line_bundle:
        shifted = torch.roll(grouped, shifts=on_data.shape[-3] // 2, dims=0)

        mask = grouped > shifted
        grouped[mask] = shifted[mask]
        labels = torch.unique(grouped)
        labels = labels[labels > 0]

        for new_label, old_label in enumerate(labels, start=1):
            grouped[grouped == old_label] = new_label

    N_components = grouped.max().item()
    return [(grouped == i + 1).float() for i in range(N_components)], N_components


def step_connected_components(
    u: torch.Tensor,
    cost: torch.Tensor,
    G_inv: tuple[float, float, float],
    dt: float,
) -> torch.Tensor:
    A1, A2, A3 = upwinds(u, select_upwind_derivative_dilation, pad_value=-1e8)
    gradient_norm = (G_inv[0] * A1**2 + G_inv[1] * A2**2 + G_inv[2] * A3**2).sqrt() / cost
    du_dt = dt * gradient_norm
    return u + du_dt


def step_distance(
    u: torch.Tensor,
    cost: torch.Tensor,
    G_inv: tuple[float, float, float],
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    A1, A2, A3 = upwinds(u, select_upwind_derivative_erosion, pad_value=1e8)
    gradient_norm = (G_inv[0] * A1**2 + G_inv[1] * A2**2 + G_inv[2] * A3**2).sqrt()
    du_dt = dt * (cost - gradient_norm)
    return u + du_dt, du_dt


def upwinds(
    u: torch.Tensor,
    selector: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pad_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward = (
        derivatives(u, pad_value)
    )

    A1 = selector(A1_forward, A1_backward)
    A2 = selector(A2_forward, A2_backward)
    A3 = selector(A3_forward, A3_backward)

    return A1, A2, A3


def derivatives(
    u: torch.Tensor,
    pad_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    Or = u.shape[0]

    (
        I_A1_forward,
        I_A1_backward,
        I_A2_forward,
        I_A2_backward,
    ) = get_A_kernels(Or, u.device)

    u_pad = F.pad(u, (1, 1, 1, 1), mode="constant", value=pad_value)

    A1_forward = F.conv2d(
        u_pad,
        I_A1_forward,
        groups=Or,
    )
    A1_backward = F.conv2d(
        u_pad,
        -I_A1_backward,
        groups=Or,
    )

    A2_forward = F.conv2d(
        u_pad,
        I_A2_forward,
        groups=Or,
    )
    A2_backward = F.conv2d(
        u_pad,
        -I_A2_backward,
        groups=Or,
    )

    dθ = 2 * torch.pi / Or

    u_padded = F.pad(
        u[None, None],
        pad=(0, 0, 0, 0, 1, 1),
        mode="circular",
    ).squeeze()

    A3_forward = (u_padded[2:] - u_padded[1:-1]) / dθ
    A3_backward = (u_padded[1:-1] - u_padded[:-2]) / dθ

    return A1_forward, A1_backward, A2_forward, A2_backward, A3_forward, A3_backward


@lru_cache
def get_A_kernels(Or: int, device: str | torch.DeviceObjType):
    θs = torch.arange(Or, device=device) * 2 * torch.pi / Or

    cos = torch.cos(θs)
    sin = torch.sin(θs)

    def bilinear_shift_kernel(dx, dy):
        k = torch.zeros(Or, 1, 3, 3, device=device)

        x_neg = (-dx).clamp(min=0)
        x_pos = dx.clamp(min=0)

        y_neg = (-dy).clamp(min=0)
        y_pos = dy.clamp(min=0)

        x_ctr = 1.0 - x_neg - x_pos
        y_ctr = 1.0 - y_neg - y_pos

        wx = torch.stack([x_neg, x_ctr, x_pos], dim=-1)
        wy = torch.stack([y_neg, y_ctr, y_pos], dim=-1)

        k[:, 0] = wy[:, :, None] * wx[:, None, :]
        k[..., 1, 1] += -1.0

        return k

    A1_forward = bilinear_shift_kernel(cos, sin)
    A1_backward = bilinear_shift_kernel(-cos, -sin)

    A2_forward = bilinear_shift_kernel(-sin, cos)
    A2_backward = bilinear_shift_kernel(sin, -cos)

    return A1_forward, A1_backward, A2_forward, A2_backward


def select_upwind_derivative_dilation(
    d_forward: torch.Tensor, d_backward: torch.Tensor
) -> torch.Tensor:
    min_d_backward = -d_backward
    maximum = torch.maximum(d_forward, min_d_backward)
    return torch.where(
        maximum > 0,
        torch.where(d_forward >= min_d_backward, -maximum, maximum),
        0.0,
    )


def select_upwind_derivative_erosion(
    d_forward: torch.Tensor, d_backward: torch.Tensor
) -> torch.Tensor:
    min_d_forward = -d_forward
    maximum = torch.maximum(min_d_forward, d_backward)
    return torch.where(
        maximum > 0,
        torch.where(min_d_forward >= d_backward, -maximum, maximum),
        0.0,
    )


def check_convergence(du_dt: torch.Tensor, boundary_condition) -> bool:
    du_dt = du_dt * boundary_condition
    return du_dt.abs().max()


def compute_timestep(
    cost: torch.Tensor,
    dθ: float,
    G_inv: tuple[float, float, float],
):
    return cost.min() / sqrt(G_inv[0] + G_inv[1] + G_inv[2] / dθ**2)
