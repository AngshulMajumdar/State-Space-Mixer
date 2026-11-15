"""Assemble the Hugging Face Space (Gradio) for the ssmix audio embedder.

    python scripts/build_space.py <output_dir>

Writes README.md (Space metadata + description, numbers from benchmarks/results.json), app.py,
requirements.txt and a copy of the ssmix package, so the Space runs without installing from GitHub.
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ssmix import AudioEmbedder, __version__  # noqa: E402

REPO = "https://github.com/AngshulMajumdar/ssmix"
DOCS = "https://angshulmajumdar.github.io/ssmix/"
GRADIO = "6.32.0"


def main(out):
    with open(os.path.join(ROOT, "benchmarks", "results.json")) as f:
        R = json.load(f)
    E, Fd = R["esc50"], R["fsdd"]
    wins = {k: sum(a > b for a, b in zip(E["ssmix"]["folds"], E[k]["folds"])) for k in ("vggish", "wav2vec2", "compare")}
    dim = AudioEmbedder().dim(5 * 32000)

    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "space", "app.py"), out)
    shutil.copy(os.path.join(ROOT, "space", "requirements.txt"), out)
    pkg = os.path.join(out, "ssmix")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    shutil.copytree(os.path.join(ROOT, "ssmix"), pkg, ignore=shutil.ignore_patterns("__pycache__", ".*"))

    readme = f"""---
title: ssmix Audio Embedder
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {GRADIO}
app_file: app.py
pinned: false
short_description: Training-free audio transformer on Daubechies wavelets
tags:
- audio
- audio-embeddings
- wavelets
- training-free
---

# ssmix audio embedder

A **training-free audio transformer**. Its token mixer is cross-scale attention on an exact Daubechies wavelet
frame. It has **zero learned parameters** and needs **no training data**.

- **Compare two clips**: upload or record two clips and get the cosine similarity of their {dim:,}-dimensional
  ssmix embeddings, plus the embeddings as a `.npy` file.
- **Inspect the scale space**: see a clip's log-mel spectrogram, its Daubechies scale space, how strongly each
  wavelet coefficient attends to its neighbours, and where each band's attention goes (parent, sibling, child).

Code and API: [{REPO.replace("https://", "")}]({REPO}) · Article: [{DOCS}article.html]({DOCS}article.html)

## How it works

1. A 128-band log-mel spectrogram (25 ms windows, 10 ms hop), normalised per clip.
2. An undecimated **Daubechies (db4) tight frame** along time, computed with one batched FFT. It is Parseval-tight,
   so reconstruction is exact, and its cost does not depend on filter support.
3. **Cross-scale attention**. Every coefficient attends to its dyadic siblings, parent and children, using
   parameter-free cosine weights. There are two blocks.
4. Readout: band magnitudes before and after mixing, the attention maps, a spectral scale space and the
   spectral envelope.

## Results (frozen embeddings, identical linear probe)

**ESC-50**, environmental sound, 5-fold: ssmix **{E['ssmix']['mean']}%**, ahead of VGGish ({E['vggish']['mean']}%, {wins['vggish']}/5 folds),
wav2vec 2.0 ({E['wav2vec2']['mean']}%, {wins['wav2vec2']}/5 folds), openSMILE ComParE ({E['compare']['mean']}%, {wins['compare']}/5 folds) and all eight
hand-crafted feature sets combined ({E['handcrafted_all']['mean']}%).

**FSDD**, spoken digits, official split: ssmix **{Fd['ssmix']}%**, ahead of EfficientAT ({Fd['efficientat_mn10']}%),
PaSST ({Fd['passt']}%), wav2vec 2.0 ({Fd['wav2vec2']}%) and PANNs ({Fd['panns']}%).

## Use it in Python

```bash
pip install git+{REPO}
```

```python
from ssmix import AudioEmbedder
E = AudioEmbedder().embed(waves, sample_rate=44100, length=5.0)   # (N, {dim}) numpy
```

ssmix {__version__}.
"""
    with open(os.path.join(out, "README.md"), "w", encoding="utf8") as f:
        f.write(readme)
    print("space assembled in", out)


if __name__ == "__main__":
    main(sys.argv[1])
