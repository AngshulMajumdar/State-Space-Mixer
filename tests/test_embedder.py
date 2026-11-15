import numpy as np
import pytest
import torch

from ssmix import AudioEmbedder


@pytest.fixture(scope="module")
def emb():
    return AudioEmbedder()


def test_no_parameters(emb):
    assert sum(p.numel() for p in emb.parameters()) == 0


def test_shape_finite_and_dim(emb):
    x = 0.1 * torch.randn(3, 32000 * 2)
    e = emb(x)
    assert e.shape == (3, emb.dim(32000 * 2))
    assert torch.isfinite(e).all()


def test_batch_invariance(emb):
    x = 0.1 * torch.randn(4, 32000)
    full = emb(x)
    single = torch.cat([emb(x[i:i + 1]) for i in range(4)])
    assert torch.allclose(full, single, atol=1e-5)


def test_embed_handles_lists_lengths_and_resampling(emb):
    rng = np.random.default_rng(0)
    clips = [0.1 * rng.standard_normal(n).astype(np.float32) for n in (44100, 2 * 44100, 30000)]
    E = emb.embed(clips, sample_rate=44100, length=2.0)
    assert E.shape == (3, emb.dim(64000))
    # order preserved: re-embedding one clip alone gives its row
    one = emb.embed(clips[1], sample_rate=44100, length=2.0)
    assert np.allclose(one[0], E[1], atol=1e-5)


def test_embed_rejects_mixed_dimensions_without_length(emb):
    rng = np.random.default_rng(1)
    clips = [rng.standard_normal(n).astype(np.float32) for n in (3200, 320000)]
    with pytest.raises(ValueError):
        emb.embed(clips)


def test_readout_validation():
    with pytest.raises(ValueError):
        AudioEmbedder(readout=("bands", "nope"))
    with pytest.raises(ValueError):
        AudioEmbedder(n_mels=100, heads=16)


def test_readout_terms_are_separable():
    x = 0.1 * torch.randn(1, 32000)
    full = AudioEmbedder(readout=("bands", "env"))(x)
    env = AudioEmbedder(readout=("env",))(x)
    assert torch.allclose(full[:, -env.shape[1]:], env)


def test_save_and_from_pretrained_round_trip(tmp_path):
    emb = AudioEmbedder(depth=1, n_mels=64, heads=8, window=1)
    emb.save_pretrained(str(tmp_path))
    back = AudioEmbedder.from_pretrained(str(tmp_path))
    assert back.config == emb.config
    x = 0.1 * torch.randn(2, 32000)
    assert torch.equal(back(x), emb(x))


def test_default_config_is_the_benchmarked_model():
    c = AudioEmbedder().config
    assert (c["depth"], c["sample_rate"], c["n_mels"], c["order"], c["heads"], c["window"], c["tau"]) == \
        (2, 32000, 128, 4, 16, 2, 0.1)
