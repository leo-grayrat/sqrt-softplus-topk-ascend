import pytest
import torch

from reference.reference import expected_output, official, stable_topk_first, topk_first


@pytest.mark.parametrize(
    ("dtype", "e", "k"),
    [
        (torch.float16, 64, 1),
        (torch.float16, 257, 6),
        (torch.float32, 65, 6),
        (torch.float32, 1024, 64),
    ],
)
def test_topk_first_matches_official_on_seeded_random_inputs(dtype, e, k):
    generator = torch.Generator().manual_seed(20260910)
    x = torch.randn((4, e), generator=generator, dtype=torch.float32).to(dtype)

    expected = official(x, k)
    actual = topk_first(x, k)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_topk_first_matches_official_on_numeric_edges(dtype):
    x = torch.tensor(
        [
            [float("-inf"), -100.0, -20.0, -1.0, 0.0, 1.0, 19.999, 20.0, 20.001, 100.0],
            [3.0, 3.0, 3.0, 2.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
        ],
        dtype=dtype,
    )

    for k in (1, 6, 10):
        assert torch.equal(topk_first(x, k), official(x, k))


def test_official_preserves_the_published_fp32_computation():
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)

    assert official(x, 1).dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_expected_output_matches_declared_output_dtype(dtype):
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)

    result = expected_output(x, 2)

    assert result.dtype == dtype
    assert torch.equal(result.float(), official(x, 2).to(dtype).float())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_stable_topk_first_matches_official_within_competition_tolerance(dtype):
    generator = torch.Generator().manual_seed(20260910)
    inputs = [
        torch.randn((8, 257), generator=generator, dtype=torch.float32).to(dtype),
        (torch.rand((8, 1024), generator=generator) * 200.0 - 100.0).to(dtype),
        torch.tensor(
            [[float("-inf"), -100.0, -20.0, -1.0, 0.0, 1.0, 19.999, 20.0, 20.001, 100.0, float("inf")]],
            dtype=dtype,
        ),
    ]
    atol = rtol = 1e-3 if dtype == torch.float16 else 1e-4

    for x in inputs:
        for k in (1, min(6, x.shape[1]), min(64, x.shape[1])):
            expected = official(x, k).to(dtype)
            actual = stable_topk_first(x, k)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
