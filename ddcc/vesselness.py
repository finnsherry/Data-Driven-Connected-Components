"""
vesselness
==========

Compute the M_2 vesselness.
"""

import torch
import torch.nn.functional as F
from math import ceil
from functools import lru_cache


def single_scale_vesselness(U, σ_s_int, σ_o_int, σ_s_ext, σ_o_ext, mask=None):
    """
    Compute the crossing-preserving vesselness[1] of the orientation score of an
    image `U`.

    References:
      [1]: J. Hannink, R. Duits, and E. Bekkers.
        "Crossing-Preserving Multi-scale Vesselness". In: Medical Image
        Computing and Computer-Assisted Intervention 8674 (2014), pp. 603-610.
        DOI:10.1007/978-3-319-10470-6_75
    """
    ## Apply internal regularisation.
    U_int = gaussian_blur_3d_separable(U, (σ_s_int, σ_s_int, σ_o_int))

    ## Compute A_11 and A_22 derivatives.
    I_A1_forward, I_A1_backward, I_A2_forward, I_A2_backward = get_A_grids(U)
    u_expanded = U_int[:, None]
    A1_forward = F.grid_sample(
        u_expanded, I_A1_forward, padding_mode="border", align_corners=True
    ).squeeze()
    A1_backward = F.grid_sample(
        u_expanded, I_A1_backward, padding_mode="border", align_corners=True
    ).squeeze()

    A2_forward = F.grid_sample(
        u_expanded, I_A2_forward, padding_mode="border", align_corners=True
    ).squeeze()
    A2_backward = F.grid_sample(
        u_expanded, I_A2_backward, padding_mode="border", align_corners=True
    ).squeeze()
    A11_U = A1_forward - 2 * U_int + A1_backward
    A22_U = A2_forward - 2 * U_int + A2_backward

    ## Apply external regularisation to derivatives.
    λ1 = gaussian_blur_3d_separable(A11_U, (σ_s_ext, σ_s_ext, σ_o_ext))
    c = gaussian_blur_3d_separable(A22_U, (σ_s_ext, σ_s_ext, σ_o_ext))

    # Combine components.
    Q = c  # Convexity criterion.
    S = torch.sqrt(λ1**2 + c**2)  # Structure measure.
    R = λ1 / (c + 1e-6)  # Anisotropy measure.

    if mask is not None:
        Q *= mask
        S *= mask
        R *= mask

    σ1 = 0.5
    σ2 = S.max()

    lineness = torch.exp(-(R**2) / (2 * σ1**2)) * (1 - torch.exp(-(S**2) / (2 * σ2**2)))
    # Vessels are dark lines, so they are locally convex. We can assess
    # local convexity by looking at the left invariant perpendicular
    # Laplacian, given by A_22.
    is_convex = Q > 0.0

    V = lineness * is_convex
    return V


def multi_scale_vesselness(U, σ_s_list, σ_o, σ_s_ext, σ_o_ext, mask=None):
    """
    Compute the multiscale vesselness of the orientation score of an image
    `U` by combining crossing-preserving vesselnesses[1] at various scales via
    maximum projection.

    Args:
        `U`: torch.Tensor of orientation score, with shape [Or, H, W].
        `σ_s_list`: standard deviations in pixels of the internal regularisation
          in the spatial directions before taking derivatives.
        `σ_o`: standard deviation in pixels of the internal regularisation
          in the orientational direction before taking derivatives.
        `σ_s_ext`: standard deviation in pixels of the external regularisation
          in the spatial direction after taking derivatives.
          Notably, this regularisation is NOT truly external, because it
          commutes with the derivatives.
        `σ_o_ext`: standard deviation in pixels of the internal regularisation
          in the orientational direction after taking derivatives.
          Notably, this regularisation is NOT truly external, because it
          commutes with the derivatives.

    Returns:
        torch.Tensor of multi scale vesselness of orientation score of retinal
          image `U`.

    References:
      [1]: J. Hannink, R. Duits, and E. Bekkers.
        "Crossing-Preserving Multi-scale Vesselness". In: Medical Image
        Computing and Computer-Assisted Intervention 8674 (2014), pp. 603-610.
        DOI:10.1007/978-3-319-10470-6_75
    """
    Vs = torch.zeros((len(σ_s_list), *U.shape), device=U.device)
    for i, σ_s in enumerate(σ_s_list):
        Vs[i] = single_scale_vesselness(U, σ_s, σ_o, σ_s_ext, σ_o_ext, mask=mask)
    V_unnormalised = Vs.max(0)[0]  # Vs.sum(0) ?
    V = (V_unnormalised - V_unnormalised.min()) / (
        V_unnormalised.max() - V_unnormalised.min()
    )
    return V


def field_of_view_mask(image, threshold=0.05, erosion=3., σ=2.):
    """Mask of the field of view of `image`, i.e. without the black padding.
    The boundary is smoothed on scale `σ` and moved `erosion` pixels inwards."""
    radius = int(4 * σ) + 1
    xs = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.float32)
    kernel = torch.exp(-xs**2 / (2 * σ**2))
    kernel = kernel / kernel.sum()
    blurred = (image > threshold).float()[None, None]
    blurred = F.conv2d(blurred, kernel[None, None, :, None], padding=(radius, 0))
    blurred = F.conv2d(blurred, kernel[None, None, None, :], padding=(0, radius))
    # Along a straight edge, the blurred step equals Φ(distance / σ).
    return blurred[0, 0] > torch.special.ndtr(torch.tensor(erosion / σ))


def gaussian_blur_3d_separable(x, sigmas, truncate=4.0):
    kernel_size = [1 + 2 * ceil(truncate * sigma) for sigma in sigmas]

    x = x[None, None]

    kx = gaussian_kernel_1d(kernel_size[0], sigmas[0], x.device, x.dtype)
    ky = gaussian_kernel_1d(kernel_size[1], sigmas[1], x.device, x.dtype)
    kz = gaussian_kernel_1d(kernel_size[2], sigmas[2], x.device, x.dtype)

    kx = kx.view(1, 1, 1, 1, -1)
    ky = ky.view(1, 1, 1, -1, 1)
    kz = kz.view(1, 1, -1, 1, 1)

    x = F.pad(x, (kernel_size[0] // 2, kernel_size[0] // 2, 0, 0, 0, 0), mode="reflect")
    x = F.conv3d(x, kx)
    x = F.pad(x, (0, 0, kernel_size[1] // 2, kernel_size[1] // 2, 0, 0), mode="reflect")
    x = F.conv3d(x, ky)
    x = F.pad(
        x, (0, 0, 0, 0, kernel_size[2] // 2, kernel_size[2] // 2), mode="circular"
    )
    x = F.conv3d(x, kz)

    return x[0, 0]


def gaussian_kernel_1d(kernel_size, sigma, device=None, dtype=torch.float32):
    coords = (
        torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    )
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


@lru_cache
def get_A_grids(u: torch.Tensor):
    Or, H, W = u.shape
    device = u.device
    θs = torch.arange(Or, device=device) * 2 * torch.pi / Or
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H).to(device),
        torch.linspace(-1, 1, W).to(device),
        indexing="ij",
    )
    dx = 2.0 / (W - 1)
    dy = 2.0 / (H - 1)

    cos = torch.cos(θs)[:, None, None]
    sin = torch.sin(θs)[:, None, None]
    I_A1_forward = torch.stack((x + cos * dx, y + sin * dy), dim=-1) * 1.0
    I_A1_backward = torch.stack((x - cos * dx, y - sin * dy), dim=-1) * 1.0
    I_A2_forward = torch.stack((x - sin * dx, y + cos * dy), dim=-1) * 1.0
    I_A2_backward = torch.stack((x + sin * dx, y - cos * dy), dim=-1) * 1.0
    return I_A1_forward, I_A1_backward, I_A2_forward, I_A2_backward
