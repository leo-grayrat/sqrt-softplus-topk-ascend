import pytest
import torch

from reference.reference import pivot_filter_reference, topk_first


@pytest.mark.parametrize(
    ("e", "k"),
    [
        (65, 2),
        (95, 4),
        (127, 8),
        (128, 4),
        (257, 8),
        (511, 4),
        (1023, 8),
        (1024, 8),
    ],
)
def test_pivot_filter_matches_topk_first_on_seeded_random_inputs(e, k):
    generator = torch.Generator().manual_seed(20260912 + e + k)
    x = torch.randn((6, e), generator=generator, dtype=torch.float32)

    actual = pivot_filter_reference(x, k)
    expected = topk_first(x, k)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_pivot_filter_matches_topk_first_with_many_duplicate_values():
    row = torch.tensor(
        [9.0] * 3 + [5.0] * 40 + [1.0] * 80 + [-3.0] * 5,
        dtype=torch.float32,
    )
    x = torch.stack([row, row.flip(0)])

    for k in (2, 4, 8):
        torch.testing.assert_close(
            pivot_filter_reference(x, k),
            topk_first(x, k),
            atol=1e-5,
            rtol=1e-5,
        )


def test_pivot_filter_fallback_remains_exact_on_all_equal_rows():
    x = torch.full((3, 257), 2.5, dtype=torch.float32)

    for k in (2, 4, 8):
        torch.testing.assert_close(
            pivot_filter_reference(x, k),
            topk_first(x, k),
            atol=1e-5,
            rtol=1e-5,
        )


def test_pivot_filter_fallback_remains_exact_with_infinities():
    x = torch.tensor(
        [
            [float("inf"), 7.0, 3.0, 1.0, 0.0, -1.0, float("-inf")],
            [5.0, 4.0, float("inf"), 2.0, float("-inf"), -3.0, -4.0],
        ],
        dtype=torch.float32,
    )

    for k in (2, 4):
        torch.testing.assert_close(
            pivot_filter_reference(x, k),
            topk_first(x, k),
            atol=1e-5,
            rtol=1e-5,
        )
