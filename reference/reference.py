import torch


def _softplus_reference(x: torch.Tensor) -> torch.Tensor:
    """Published stable softplus branch used by the competition reference."""
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))


def official(router_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mirror the published PyTorch reference computation.

    The published code calls ``router_logits.float()`` and does not cast the
    result back, so this function returns FP32 even for FP16 input.
    """
    scores = torch.sqrt(_softplus_reference(router_logits.float()))
    return torch.sort(scores, dim=-1, descending=True).values[:, :k]


def topk_first(router_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Select logits first, then transform only the selected k values."""
    logits = torch.topk(
        router_logits,
        k,
        dim=-1,
        largest=True,
        sorted=True,
    ).values
    return torch.sqrt(_softplus_reference(logits.float()))


def hierarchical_topk_first(
    router_logits: torch.Tensor,
    k: int,
    chunk: int = 32,
    candidate_alignment: int | None = None,
) -> torch.Tensor:
    """每个 32 元素块先取局部 top-k，再从候选中取全局 top-k。

    ``candidate_alignment`` 用于模拟 Ascend TopK NORMAL 对 inner 为 32
    的整数倍的要求：局部候选在行尾补 -inf 后再做最终 TopK。
    """
    pad = (-router_logits.shape[-1]) % chunk
    logits = router_logits
    if pad:
        logits = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
    chunks = logits.reshape(*logits.shape[:-1], -1, chunk)
    local = torch.topk(chunks, k, dim=-1, largest=True, sorted=True).values
    candidates = local.flatten(-2)
    if candidate_alignment:
        candidate_pad = (-candidates.shape[-1]) % candidate_alignment
        if candidate_pad:
            candidates = torch.nn.functional.pad(candidates, (0, candidate_pad), value=float("-inf"))
    selected = torch.topk(candidates, k, dim=-1, largest=True, sorted=True).values
    x = selected.float()
    softplus = torch.clamp_min(x, 0.0) + torch.log1p(torch.exp(-torch.abs(x)))
    return torch.sqrt(softplus).to(router_logits.dtype)


def stable_topk_first(router_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k first, then use a branchless stable softplus form in FP32.

    softplus(x) = max(x, 0) + log(1 + exp(-abs(x)))
    """
    logits = torch.topk(
        router_logits,
        k,
        dim=-1,
        largest=True,
        sorted=True,
    ).values
    x = logits.float()
    softplus = torch.clamp_min(x, 0.0) + torch.log1p(torch.exp(-torch.abs(x)))
    return torch.sqrt(softplus).to(router_logits.dtype)


def pivot_filter_reference(
    router_logits: torch.Tensor,
    k: int,
    candidate_limit: int = 64,
    max_iters: int = 8,
) -> torch.Tensor:
    """Exact pivot-filter model used by the Ascend rewrite.

    Each row is narrowed only when the threshold relation proves that the
    candidate set contains every global top-k value. If no safe set no larger
    than ``candidate_limit`` is found within ``max_iters``, the row falls back
    to exact top-k.
    """
    if router_logits.ndim != 2:
        raise ValueError("pivot_filter_reference expects a 2D tensor")
    if k <= 0 or k > router_logits.shape[1]:
        raise ValueError("k must be in [1, E]")

    selected_rows = []
    for row in router_logits:
        candidate = None

        # +/-inf makes arithmetic midpoint selection ill-defined. These are
        # correctness edges rather than the performance target, so deliberately
        # exercise the exact fallback contract.
        if torch.isfinite(row).all():
            lo = float(torch.min(row).item())
            hi = float(torch.max(row).item())

            for _ in range(max_iters):
                if not lo < hi:
                    break
                pivot = lo + (hi - lo) * 0.5
                gt_mask = row > pivot
                ge_mask = row >= pivot
                count_gt = int(gt_mask.sum().item())
                count_ge = int(ge_mask.sum().item())

                if count_gt >= k:
                    if count_gt <= candidate_limit:
                        candidate = row[gt_mask]
                        break
                    lo = pivot
                    continue

                if count_ge >= k:
                    if count_ge <= candidate_limit:
                        candidate = row[ge_mask]
                    break

                hi = pivot

        if candidate is None:
            chosen = torch.topk(row, k, largest=True, sorted=True).values
        else:
            chosen = torch.sort(candidate, descending=True).values[:k]
        selected_rows.append(chosen)

    logits = torch.stack(selected_rows, dim=0)
    return torch.sqrt(_softplus_reference(logits.float()))


def expected_output(router_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply the declared output-dtype contract to the published computation."""
    return official(router_logits, k).to(router_logits.dtype)
