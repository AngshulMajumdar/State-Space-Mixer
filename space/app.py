"""ssmix: a training-free audio transformer on Daubechies wavelets (Hugging Face Space)."""
import os
import tempfile
import wave

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from ssmix import AudioEmbedder  # noqa: E402

torch.set_grad_enabled(False)
EMB = AudioEmbedder()  # zero parameters: nothing to download
SR = EMB.sample_rate
CLIP_SECONDS = 5.0
MAX_SECONDS = 30.0


# ------------------------------------------------------------------ audio helpers
def to_wave(audio):
    """Gradio (sample_rate, ndarray) -> float32 mono at SR, clipped to MAX_SECONDS."""
    if audio is None:
        raise gr.Error("Please upload or record a clip.")
    sr, x = audio
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.size == 0:
        raise gr.Error("The clip is empty.")
    t = torch.from_numpy(x)
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    return t[: int(MAX_SECONDS * SR)]


def fixed_length(t, seconds=CLIP_SECONDS):
    n = int(seconds * SR)
    return torch.nn.functional.pad(t, (0, max(0, n - t.numel())))[:n]


# ------------------------------------------------------------------ compare tab
def compare(a, b):
    ta, tb = fixed_length(to_wave(a)), fixed_length(to_wave(b))
    E = EMB(torch.stack([ta, tb])).numpy()
    Z = E / np.linalg.norm(E, axis=1, keepdims=True)
    cos = float(Z[0] @ Z[1])
    path = os.path.join(tempfile.mkdtemp(), "ssmix_embeddings.npy")
    np.save(path, E.astype(np.float32))
    msg = (f"### Cosine similarity: **{cos:.3f}**\n"
           f"Each clip is padded or cut to {CLIP_SECONDS:.0f} s and mapped to a {E.shape[1]:,}-dimensional "
           f"embedding with zero learned parameters. For classification, use the embeddings with a linear model "
           f"or k-NN; raw cosine is a quick, unsupervised view.")
    return msg, path


# ------------------------------------------------------------------ inspect tab
def inspect(a):
    x = fixed_length(to_wave(a))
    X = torch.log(EMB.mel(x.unsqueeze(0)).transpose(1, 2) + 1e-6)  # (1, T, F)
    Xc = X - X.mean(1, keepdim=True)
    Wp, plan, T = EMB.mixer.coefficients(Xc)
    P = plan["P"]
    EMB.mixer.mix(Wp)
    att = [a_[..., P:P + T] for a_ in EMB.mixer.last_attention]  # each (1, M_s, H, T)
    links = EMB.mixer.links(Wp.shape[1])
    S = Wp.shape[1]
    band_names = [f"D{j + 1}" for j in range(S - 1)] + [f"A{S - 1}"]
    t_axis = np.arange(T) * 0.01

    fig, ax = plt.subplots(4, 1, figsize=(9, 11), gridspec_kw={"height_ratios": [1.2, 1, 1, 1]})
    ax[0].imshow(X[0].T.numpy(), origin="lower", aspect="auto", cmap="magma", extent=[0, t_axis[-1], 0, X.shape[2]])
    ax[0].set_title("Log-mel spectrogram (128 bands, 32 kHz)")
    ax[0].set_ylabel("mel band")

    E = Wp[0, :, :, P:P + T].abs().mean(1).numpy()  # (S, T)
    E = E / (E.max(axis=1, keepdims=True) + 1e-12)
    ax[1].imshow(E, origin="lower", aspect="auto", cmap="viridis", extent=[0, t_axis[-1], -0.5, S - 0.5])
    ax[1].set_yticks(range(S), band_names)
    ax[1].set_title("Daubechies (db4) scale space along time: band magnitude, normalised per band")

    mixing = np.stack([1.0 - a_[0, 0].mean(0).numpy() for a_ in att])  # (S, T): weight given away from self
    im = ax[2].imshow(mixing, origin="lower", aspect="auto", cmap="cividis", vmin=0, vmax=max(mixing.max(), 1e-3),
                      extent=[0, t_axis[-1], -0.5, S - 0.5])
    ax[2].set_yticks(range(S), band_names)
    ax[2].set_title("Cross-scale attention: weight each coefficient takes from its neighbours (1 − self weight)")
    ax[2].set_xlabel("time (s)")
    fig.colorbar(im, ax=ax[2], fraction=0.02, pad=0.01)

    kinds = {"child (finer)": [], "sibling": [], "parent (coarser)": []}
    for s, ks in enumerate(links):
        w = att[s][0].mean(-1).mean(-1).numpy()  # (M,)
        acc = {k: 0.0 for k in kinds}
        for (s2, sh), v in zip(ks, w):
            if s2 == s and sh == 0:
                continue
            acc["sibling" if s2 == s else ("child (finer)" if s2 < s else "parent (coarser)")] += float(v)
        for k in kinds:
            kinds[k].append(acc[k])
    left = np.zeros(S)
    for k, c in zip(kinds, ("#4c78a8", "#9ecae9", "#f58518")):
        ax[3].barh(range(S), kinds[k], left=left, color=c, label=k)
        left += np.array(kinds[k])
    ax[3].set_yticks(range(S), band_names)
    ax[3].set_xlabel("mean attention weight")
    ax[3].set_title("Where each band's attention goes (excluding itself)")
    ax[3].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ examples (synthetic, no licences)
def _write(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    return path


def make_examples():
    d = tempfile.mkdtemp()
    t = np.arange(int(4 * 16000)) / 16000
    rng = np.random.default_rng(0)
    tone = sum(np.sin(2 * np.pi * k * 220 * t) / k for k in range(1, 6)) * 0.15
    tone2 = sum(np.sin(2 * np.pi * k * 330 * t) / k for k in range(1, 6)) * 0.15
    chirp = 0.3 * np.sin(2 * np.pi * (100 * t + 400 * t ** 2))
    bursts = np.zeros_like(t)
    for c in rng.uniform(0, 3.8, size=8):
        i = int(c * 16000)
        bursts[i:i + 1200] = rng.standard_normal(len(bursts[i:i + 1200])) * 0.4
    return {n: _write(os.path.join(d, f"{n}.wav"), s) for n, s in
            (("tone_220Hz", tone), ("tone_330Hz", tone2), ("chirp", chirp), ("noise_bursts", bursts))}


EX = make_examples()

ABOUT = """
# ssmix · training-free audio transformer
Daubechies wavelets + cross-scale attention, with **zero learned parameters** and **no training data**.
Each wavelet coefficient attends to its dyadic siblings, parent and children. The embedding is read out from the
scale space and from the attention maps themselves.
[Code](https://github.com/AngshulMajumdar/ssmix) ·
[Article](https://angshulmajumdar.github.io/ssmix/article.html) ·
[Docs](https://angshulmajumdar.github.io/ssmix/)
"""

with gr.Blocks(title="ssmix: training-free audio transformer") as demo:
    gr.Markdown(ABOUT)
    with gr.Tab("Compare two clips"):
        with gr.Row():
            a = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Clip A")
            b = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Clip B")
        go = gr.Button("Embed and compare", variant="primary")
        out = gr.Markdown()
        f = gr.File(label="Embeddings (.npy, shape 2 × D)")
        go.click(compare, [a, b], [out, f])
        gr.Examples([[EX["tone_220Hz"], EX["tone_330Hz"]], [EX["tone_220Hz"], EX["noise_bursts"]],
                     [EX["chirp"], EX["noise_bursts"]]], inputs=[a, b])
    with gr.Tab("Inspect the scale space"):
        c = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Clip")
        go2 = gr.Button("Analyse", variant="primary")
        plot = gr.Plot()
        go2.click(inspect, c, plot)
        gr.Examples([[EX["chirp"]], [EX["noise_bursts"]], [EX["tone_220Hz"]]], inputs=[c])

if __name__ == "__main__":
    demo.launch()
