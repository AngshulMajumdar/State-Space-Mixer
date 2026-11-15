"""ssmix quickstart: training-free audio embeddings and the mixers as PyTorch layers.

    python examples/quickstart.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssmix import AudioEmbedder, ScaleSpaceMixer, SWTCrossScaleMixer  # noqa: E402

rng = np.random.default_rng(0)
sr = 16000
t = np.arange(2 * sr) / sr


def tone():  # harmonic tone with random pitch
    f0 = rng.uniform(200, 400)
    return sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6)) * 0.1


def bursts():  # trains of short noise bursts
    x = np.zeros_like(t)
    for c in rng.uniform(0, 1.9, size=6):
        i = int(c * sr)
        x[i:i + 800] = rng.standard_normal(800) * 0.3
    return x


# 1) Embeddings: no training, any sample rate, any clip length (fixed with `length`)
clips = [tone() for _ in range(10)] + [bursts() for _ in range(10)]
labels = np.array([0] * 10 + [1] * 10)
emb = AudioEmbedder()
E = emb.embed(clips, sample_rate=sr, length=2.0)
print("embeddings:", E.shape)

# leave-one-out 1-nearest-neighbour on standardised embeddings
Z = (E - E.mean(0)) / (E.std(0) + 1e-8)
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
S = Z @ Z.T
np.fill_diagonal(S, -np.inf)
print("1-NN leave-one-out accuracy:", (labels[S.argmax(1)] == labels).mean())

# 2) The mixers are ordinary nn.Modules on (batch, tokens, channels)
x = torch.randn(2, 1000, 64)
frame = ScaleSpaceMixer(order=4, rule="identity")       # exact tight-frame round trip
print("tight-frame reconstruction error:", (frame(x) - x).abs().max().item())
W = frame.analysis(x)                                     # (2, J+1, 1000, 64) scale-space coefficients
print("scale-space coefficients:", tuple(W.shape))
mixer = SWTCrossScaleMixer(order=4, heads=8, window=2, tau=0.1)
y = mixer(x)                                              # cross-scale attention, same shape as x
print("cross-scale mixer output:", tuple(y.shape), "parameters:", sum(p.numel() for p in mixer.parameters()))
