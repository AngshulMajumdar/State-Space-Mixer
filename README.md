# ssmix

**Training-free scale-space transformers on Daubechies wavelets.**

ssmix replaces learned attention with exact multiscale operators:

- Daubechies wavelets of any order.
- A GPU-batched tight frame that computes every scale in one FFT.
- A cross-scale attention mixer in which each wavelet coefficient attends to its siblings, parent and children.

Nothing is trained, and there are no weights to download.

- Documentation and article: **https://angshulmajumdar.github.io/ssmix/**
- Live demo (Hugging Face Space): **https://huggingface.co/spaces/Angshul/ssmix-audio-embedder**

| | |
|---|---|
| ESC-50, linear probe on frozen embeddings | **74.6 % with no training**: ahead of VGGish (71.0), wav2vec 2.0 (68.4) and openSMILE ComParE (71.2) |
| FSDD spoken digits, official split | **99.0 %**: ahead of EfficientAT (98.7), PaSST (98.0), wav2vec 2.0 (97.0) and PANNs (85.7) |
| Reconstruction error of the frame and the DWT | 10⁻¹² (float64) |
| Filters | identical to PyWavelets db1–db38; orthonormal with N vanishing moments, verified to db60 |
| Learnable parameters | 0 |

## Install

```bash
pip install git+https://github.com/AngshulMajumdar/ssmix
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, torchaudio, NumPy and mpmath.

## Audio embeddings

```python
from ssmix import AudioEmbedder

emb = AudioEmbedder()                                          # parameter-free; the benchmarked configuration
E = emb.embed(waves, sample_rate=44100, length=5.0)            # list of 1-D arrays -> (N, D) numpy
```

## Mixers as PyTorch layers

```python
import torch
from ssmix import SWTCrossScaleMixer, ScaleSpaceMixer, DWT

x = torch.randn(8, 4096, 64)                                   # (batch, tokens, channels)
y = SWTCrossScaleMixer(order=4, heads=8, window=2, tau=0.1)(x)  # cross-scale attention

frame = ScaleSpaceMixer(order=8, rule="identity")              # db8 tight frame, support 15
W = frame.analysis(x)                                          # (8, J+1, 4096, 64)

ds, a = DWT(order=4)(x.transpose(1, 2), J=5)                   # PyWavelets 'periodization' DWT, batched
```

The support is set by `order` (N vanishing moments, support 2N−1), or directly with `support=`.

## Public API

| Object | Role |
|---|---|
| `AudioEmbedder` | Training-free audio transformer: log-mel → cross-scale mixer blocks → pooled scale-space readout; `embed`, `from_pretrained`, `save_pretrained` |
| `SWTCrossScaleMixer` | Cross-scale attention on the undecimated, delay-aligned Daubechies frame (shift-equivariant) |
| `CrossScaleMixer` | Cross-scale attention on the decimated orthonormal wavelet tree |
| `ScaleSpaceMixer` | FFT-batched Daubechies tight frame for any length, with parameter-free gate rules |
| `DWT` | Periodic orthonormal DWT / IDWT, identical to PyWavelets `periodization` |
| `daubechies`, `daubechies_lowpass`, `qmf_highpass` | Daubechies filters of any order (spectral factorisation, mpmath roots) |

## Repository layout

```
ssmix/         package
tests/         pytest suite (exactness, equivalence with PyWavelets, mixer properties, API)
benchmarks/    prep.py, embed.py, probe.py, report.py, speed.py, results.json
examples/      quickstart.py
space/         Gradio app for the Hugging Face Space (app.py, requirements.txt)
scripts/       build_docs.py (regenerates docs/), build_space.py (assembles the Space)
docs/          GitHub Pages site with the article (Settings -> Pages -> Deploy from branch, folder /docs)
```

## Tests

```bash
pip install pytest PyWavelets
pytest tests
```

## Reproducing the benchmarks

```bash
git clone https://github.com/karolpiczak/ESC-50 data/ESC-50
git clone https://github.com/Jakobovski/free-spoken-digit-dataset data/fsdd
pip install scikit-learn pandas soundfile kymatio opensmile spafe librosa resampy transformers hear21passt panns_inference
python benchmarks/prep.py
python benchmarks/embed.py esc50 ssmix logmel mfcc scattering random_transformer egemaps compare librosa gfcc cqt handcrafted_all vggish wav2vec2
python benchmarks/embed.py fsdd  ssmix logmel mfcc scattering random_transformer librosa gfcc cqt egemaps vggish wav2vec2 passt efficientat_mn10 panns
python benchmarks/report.py
```

Pretrained baselines need their weights in `weights/` and their code in `ext/`; the header of `benchmarks/embed.py`
lists the exact files and sources.
