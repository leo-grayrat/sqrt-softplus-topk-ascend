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


def hierarchical_topk_first(router_logits: torch.Tensor, k: int, chunk: int = 32) -> torch.Tensor:
    """每个 32 元素块先取局部 top-k，再从候选中取全局 top-k。"""
    pad = (-router_logits.shape[-1]) % chunk
    logits = router_logits
    if pad:
        logits = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
    chunks = logits.reshape(*logits.shape[:-1], -1, chunk)
    local = torch.topk(chunks, k, dim=-1, largest=True, sorted=True).values
    selected = torch.topk(local.flatten(-2), k, dim=-1, largest=True, sorted=True).values
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


def expected_output(router_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply the declared output-dtype contract to the published computation."""
    return official(router_logits, k).to(router_logits.dtype)
