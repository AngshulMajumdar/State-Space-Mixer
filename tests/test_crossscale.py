import numpy as np
import pytest
import torch

from ssmix.crossscale import DWT, CrossScaleMixer, SWTCrossScaleMixer


@pytest.mark.parametrize("N,J", [(1, 6), (2, 7), (3, 4), (4, 5), (10, 3), (16, 2)])
def test_dwt_matches_pywavelets_periodization(N, J):
    pywt = pytest.importorskip("pywt")
    x = torch.randn(3, 4, 256, dtype=torch.float64)
    ds, a = DWT(N)(x, J)
    ref = pywt.wavedec(x[1, 2].numpy(), f"db{N}", mode="periodization", level=J)
    ours = [a[1, 2].numpy()] + [d[1, 2].numpy() for d in reversed(ds)]
    assert max(np.abs(r - o).max() for r, o in zip(ref, ours)) < 1e-12


@pytest.mark.parametrize("N,J", [(1, 6), (4, 5), (8, 3), (20, 2)])
def test_dwt_perfect_reconstruction_and_parseval(N, J):
    d = DWT(N)
    x = torch.randn(2, 3, 512, dtype=torch.float64)
    ds, a = d(x, J)
    assert (d.inverse(ds, a) - x).abs().max() < 1e-12
    e = sum(t.pow(2).sum() for t in ds) + a.pow(2).sum()
    assert abs(e / x.pow(2).sum() - 1) < 1e-12


@pytest.mark.parametrize("cls", [CrossScaleMixer, SWTCrossScaleMixer])
def test_mixers_identity_limit_homogeneity_no_params(cls):
    x = torch.randn(2, 501, 64, dtype=torch.float64)
    m0 = cls(order=4, heads=8, tau=1e-4)
    assert (m0(x) - x).abs().max() < 1e-10  # tau -> 0: self wins -> exact reconstruction
    m = cls(order=4, heads=8, tau=0.1)
    assert sum(p.numel() for p in m.parameters()) == 0
    assert torch.allclose(m(2.5 * x), 2.5 * m(x), atol=1e-10)
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_swt_attention_is_a_distribution_over_cross_scale_links():
    x = torch.randn(1, 300, 16, dtype=torch.float64)
    m = SWTCrossScaleMixer(order=3, heads=4, window=2, tau=0.1)
    m(x)
    S = len(m.last_attention)
    for s, a in enumerate(m.last_attention):
        n_scales = 1 + (s > 0) + (s < S - 1)
        assert a.shape[1] == 5 * n_scales  # (2k+1) offsets per linked scale, self included once
        assert torch.allclose(a.sum(1), torch.ones_like(a.sum(1)))


def test_swt_mixer_is_shift_equivariant_in_circular_mode():
    x = torch.randn(1, 256, 8, dtype=torch.float64)
    m = SWTCrossScaleMixer(order=2, heads=2, window=1, tau=0.2, levels=4, pad="circular")
    assert torch.allclose(m(torch.roll(x, 37, 1)), torch.roll(m(x), 37, 1), atol=1e-10)
