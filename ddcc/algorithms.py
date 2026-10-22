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
    for δ in tqdm(δs):
        on_data_grown = u <= δ
        _, N_components = label_components(on_data_grown, on_data, projective_line_bundle=projective_line_bundle)
        alive_components.append(N_components)

    return u, alive_components


# Evaluation


def crossing_mask(gt: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """
    Find pixels near crossings in ground truth labels `gt`, i.e. pixels for
    which at least two distinct foreground labels lie within a square
    neighbourhood of size `2 * radius + 1`.

    Args:
        `gt`: integer torch.Tensor of shape `[H, W]` of labels, where 0 is
          background.
      Optional:
        `radius`: radius of the neighbourhood. Defaults to 2.

    Returns:
        Boolean torch.Tensor of shape `[H, W]`, True near crossings.
    """
    labels = gt[None, None].float()
    size = 2 * radius + 1
    max_label = F.max_pool2d(labels, size, stride=1, padding=radius)
    # Minimum over foreground labels: ignore background by setting it to inf.
    min_label = -F.max_pool2d(
        -torch.where(labels > 0, labels, torch.inf), size, stride=1, padding=radius
    )
    return ((max_label > 0) & (max_label != min_label))[0, 0]


def project_components(
    components: torch.Tensor,
    score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project components on M_2 to the image plane, by taking at each pixel the
    label at the orientation with largest `score` among labelled orientations.

    Args:
        `components`: integer torch.Tensor of shape `[Or, H, W]` of labels,
          where 0 is background, e.g. as returned by `connected_components`.
        `score`: torch.Tensor of shape `[Or, H, W]` used to select the
          orientation, e.g. the orientation score or vesselness.

    Returns:
        Tuple of integer torch.Tensor of shape `[H, W]` of projected labels,
        and boolean torch.Tensor of shape `[H, W]` which is True at pixels with
        multiple distinct labels across orientations.
    """
    is_labelled = components > 0
    masked_score = torch.where(is_labelled, score, -torch.inf)
    orientation = masked_score.argmax(0, keepdim=True)
    projected = components.gather(0, orientation)[0] * is_labelled.any(0)

    max_label = components.amax(0)
    min_label = torch.where(is_labelled, components, components.max() + 1).amin(0)
    is_multilabel = is_labelled.any(0) & (max_label != min_label)
    return projected, is_multilabel


def compare_components(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
    matching: Literal["iou", "hungarian"] = "iou",
    iou_threshold: float = 0.5,
    min_fraction: float = 0.1,
) -> dict:
    """
    Compare predicted components `pred` with ground truth components `gt`.

    All statistics are computed from the contingency table, which counts the
    pixels shared by each pair of labels. Pixels are considered if they lie in
    `mask` and are foreground in either `pred` or `gt`.

    Args:
        `pred`: integer torch.Tensor of shape `[H, W]` of predicted labels,
          where 0 is background.
        `gt`: integer torch.Tensor of shape `[H, W]` of ground truth labels,
          where 0 is background.
      Optional:
        `mask`: boolean torch.Tensor of shape `[H, W]` of pixels to evaluate,
          e.g. `~crossing_mask(gt)`. Defaults to all pixels.
        `matching`: how to match predicted and ground truth labels. "iou"
          matches pairs with IoU > `iou_threshold`, which is a one-to-one
          matching if `iou_threshold >= 0.5`.[1] "hungarian" matches to
          maximise the total IoU. Defaults to "iou".
        `iou_threshold`: IoU above which pairs are matched. Also used as minimum
          IoU for "hungarian" matching. Defaults to 0.5.
        `min_fraction`: fraction of a component that another label must cover
          to count towards a split or merge. Defaults to 0.1.

    Returns:
        dict with
          `accuracy`: fraction of pixels in matched pairs, among pixels in the
            foreground of either labelling.
          `grouping_accuracy`: as `accuracy`, but among pixels in the foreground
            of both labellings, so that it only measures grouping errors.
          `missed`: number of pixels foreground in `gt` but not in `pred`.
          `spurious`: number of pixels foreground in `pred` but not in `gt`.
          `VI_split`, `VI_merge`: conditional entropies H(pred | gt) and
            H(gt | pred) on pixels in the foreground of both labellings, which
            together form the variation of information.[2] `VI_split`
            measures how much ground truth components are broken up, and
            `VI_merge` how much predicted components mix ground truth
            components.
          `N_splits`: number of ground truth components which are covered by
            at least two predicted labels.
          `N_merges`: number of predicted components which cover at least two
            ground truth labels.
          `N_pred`, `N_gt`: numbers of predicted and ground truth labels.
          `matches`: list of matched pairs `(pred_label, gt_label, iou)`.
          `unmatched_pred`, `unmatched_gt`: lists of unmatched labels.

    References:
        [1]: A. Kirillov, K. He, R. Girshick, C. Rother, and P. Dollár.
          "Panoptic Segmentation." In: IEEE/CVF Conference on Computer Vision
          and Pattern Recognition (2019).
          DOI:10.1109/CVPR.2019.00963.
        [2]: M. Meilă. "Comparing Clusterings—an Information Based Distance."
          In: Journal of Multivariate Analysis (2007).
          DOI:10.1016/j.jmva.2006.11.013.
    """
    if mask is None:
        mask = torch.ones_like(gt, dtype=torch.bool)
    in_domain = mask & ((pred > 0) | (gt > 0))
    pred_labels, table, gt_labels = contingency_table(pred[in_domain], gt[in_domain])
    table = table.double()
    N_pixels = table.sum()

    # Background is label 0, which is the first row/column if present.
    i0 = int(len(pred_labels) > 0 and pred_labels[0] == 0)
    j0 = int(len(gt_labels) > 0 and gt_labels[0] == 0)
    missed = table[0, j0:].sum() if i0 else 0.
    spurious = table[i0:, 0].sum() if j0 else 0.
    # Sizes of components include pixels overlapping with background.
    pred_sizes = table.sum(1)[i0:]
    gt_sizes = table.sum(0)[j0:]
    table = table[i0:, j0:]
    pred_labels = pred_labels[i0:]
    gt_labels = gt_labels[j0:]

    iou = table / (pred_sizes[:, None] + gt_sizes[None] - table)

    match matching:
        case "iou":
            pred_idx, gt_idx = torch.nonzero(iou > iou_threshold, as_tuple=True)
        case "hungarian":
            from scipy.optimize import linear_sum_assignment

            pred_idx, gt_idx = linear_sum_assignment(iou.cpu().numpy(), maximize=True)
            pred_idx = torch.as_tensor(pred_idx, device=iou.device)
            gt_idx = torch.as_tensor(gt_idx, device=iou.device)
            keep = iou[pred_idx, gt_idx] > iou_threshold
            pred_idx, gt_idx = pred_idx[keep], gt_idx[keep]
        case _:
            raise ValueError(
                f"{matching} is not a valid matching! Choose one of `iou`, `hungarian`."
            )
    matched_pixels = table[pred_idx, gt_idx].sum()
    N_overlap = table.sum()

    # Variation of information on pixels which are foreground in both.
    p = table / N_overlap
    p_pred = p.sum(1, keepdim=True)
    p_gt = p.sum(0, keepdim=True)
    nonzero = p > 0
    VI_split = (p * torch.log(p_gt / p))[nonzero].sum()
    VI_merge = (p * torch.log(p_pred / p))[nonzero].sum()

    N_splits = ((table / gt_sizes[None] >= min_fraction).sum(0) >= 2).sum()
    N_merges = ((table / pred_sizes[:, None] >= min_fraction).sum(1) >= 2).sum()

    matched_pred = set(pred_idx.tolist())
    matched_gt = set(gt_idx.tolist())
    return {
        "accuracy": (matched_pixels / N_pixels).item(),
        "grouping_accuracy": (matched_pixels / N_overlap).item(),
        "missed": int(missed),
        "spurious": int(spurious),
        "VI_split": VI_split.item(),
        "VI_merge": VI_merge.item(),
        "N_splits": N_splits.item(),
        "N_merges": N_merges.item(),
        "N_pred": len(pred_labels),
        "N_gt": len(gt_labels),
        "matches": [
            (pred_labels[i].item(), gt_labels[j].item(), iou[i, j].item())
            for i, j in zip(pred_idx.tolist(), gt_idx.tolist())
        ],
        "unmatched_pred": [
            l.item() for i, l in enumerate(pred_labels) if i not in matched_pred
        ],
        "unmatched_gt": [
            l.item() for i, l in enumerate(gt_labels) if i not in matched_gt
        ],
    }


# Helper Functions


def contingency_table(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Count co-occurrences of labels in `a` and `b`, which have the same shape.

    Returns:
        Tuple of sorted unique labels in `a`, table of shape `[N_a, N_b]` of
        counts, and sorted unique labels in `b`.
    """
    a_labels, a_idx = torch.unique(a, return_inverse=True)
    b_labels, b_idx = torch.unique(b, return_inverse=True)
    N_a, N_b = len(a_labels), len(b_labels)
    table = torch.bincount((a_idx * N_b + b_idx).flatten(), minlength=N_a * N_b)
    return a_labels, table.reshape(N_a, N_b), b_labels


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
    return grouped, N_components


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
