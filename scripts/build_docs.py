"""Build the static documentation site in docs/ (served by GitHub Pages from /docs).

    python scripts/build_docs.py

The API reference is generated from the live signatures and docstrings of the
public objects in `ssmix.__all__`; the other pages are written here.
"""
import html
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ssmix  # noqa: E402

DOCS = os.path.join(ROOT, "docs")
REPO = "https://github.com/AngshulMajumdar/ssmix"

PAGES = [("index.html", "Overview"), ("article.html", "Article"), ("method.html", "Method"), ("api.html", "API"),
         ("benchmarks.html", "Benchmarks")]

CSS = """
:root { --ink:#1b1f24; --muted:#5b6470; --line:#e3e6ea; --bg:#ffffff; --soft:#f6f7f9; --accent:#0b6e69; --code:#f3f5f7; }
* { box-sizing: border-box; }
html { font-size: 16px; }
body { margin:0; color:var(--ink); background:var(--bg);
       font: 1rem/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
header { border-bottom:1px solid var(--line); background:var(--bg); position:sticky; top:0; z-index:10; }
.bar { max-width:920px; margin:0 auto; padding:0.9rem 1.25rem; display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap; }
.brand { font-weight:700; letter-spacing:0.01em; color:var(--ink); text-decoration:none; font-size:1.1rem; }
.brand span { color:var(--accent); }
nav { display:flex; flex:1; flex-wrap:wrap; align-items:center; }
nav a { color:var(--muted); text-decoration:none; margin-right:1.1rem; font-size:0.95rem; }
nav a.on, nav a:hover { color:var(--accent); }
nav a.gh { margin-left:auto; }
main { max-width:920px; margin:0 auto; padding:2rem 1.25rem 4rem; }
h1 { font-size:2rem; line-height:1.25; margin:0.5rem 0 0.75rem; }
h2 { font-size:1.35rem; margin:2.4rem 0 0.6rem; padding-top:0.4rem; border-top:1px solid var(--line); }
h3 { font-size:1.08rem; margin:1.6rem 0 0.4rem; }
p, li { color:var(--ink); }
.lead { font-size:1.12rem; color:var(--muted); max-width:44rem; }
a { color:var(--accent); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size:0.88rem; }
code { background:var(--code); padding:0.1rem 0.3rem; border-radius:4px; }
pre { background:var(--code); padding:0.9rem 1rem; border-radius:8px; overflow-x:auto; line-height:1.5; }
pre code { background:none; padding:0; }
table { border-collapse:collapse; width:100%; margin:0.8rem 0 1.2rem; font-size:0.95rem; }
th, td { text-align:left; padding:0.45rem 0.6rem; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; background:var(--soft); }
td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
tr.ours td { font-weight:600; background:#eef7f6; }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:0.9rem; margin:1.4rem 0; }
.card { border:1px solid var(--line); border-radius:10px; padding:0.9rem 1rem; background:var(--soft); }
.card b { display:block; margin-bottom:0.2rem; }
.card .big { font-size:1.6rem; font-weight:700; color:var(--accent); }
.note { color:var(--muted); font-size:0.92rem; }
.api-item { border:1px solid var(--line); border-radius:10px; padding:0.2rem 1.1rem 0.8rem; margin:1.2rem 0; }
.api-item h3 { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:1rem; }
.sig { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:0.85rem; color:var(--muted);
       white-space:pre-wrap; word-break:break-word; margin:0 0 0.6rem; }
.doc { white-space:pre-wrap; background:none; padding:0; margin:0.2rem 0 0.6rem; font-size:0.86rem; }
.method { margin-left:0.6rem; padding-left:0.9rem; border-left:3px solid var(--line); }
.toc { columns:2; font-size:0.95rem; }
.paper .authors { color:var(--muted); margin:-0.2rem 0 1.2rem; }
.abstract { background:var(--soft); border:1px solid var(--line); border-radius:10px; padding:1rem 1.2rem; margin:1rem 0 1.5rem; }
.caption { color:var(--muted); font-size:0.9rem; margin-top:-0.6rem; }
ol.refs li { font-size:0.92rem; margin-bottom:0.35rem; }
footer { border-top:1px solid var(--line); color:var(--muted); font-size:0.85rem; }
footer .bar { justify-content:space-between; }
@media (max-width:640px) { h1 { font-size:1.6rem; } .toc { columns:1; } nav a.gh { margin-left:0; } }
"""

MATHJAX = ('<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]]}};</script>\n'
           '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>')


def page(fname, title, body, math=False):
    nav = "".join(f'<a href="{f}"{" class=on" if f == fname else ""}>{t}</a>' for f, t in PAGES)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · ssmix</title>
<meta name="description" content="ssmix: training-free scale-space transformers on Daubechies wavelets.">
<link rel="stylesheet" href="style.css">
{MATHJAX if math else ""}
</head>
<body>
<header><div class="bar"><a class="brand" href="index.html">ss<span>mix</span></a>
<nav>{nav}<a class="gh" href="{REPO}">GitHub</a></nav></div></header>
<main>
{body}
</main>
<footer><div class="bar"><span>ssmix {ssmix.__version__} · Angshul Majumdar</span><a href="{REPO}">{REPO.replace("https://", "")}</a></div></footer>
</body>
</html>
"""


# --------------------------------------------------------------------- API page
def sig(obj, name):
    try:
        try:
            s = str(inspect.signature(obj, eval_str=True))
        except NameError:
            s = str(inspect.signature(obj))
    except (TypeError, ValueError):
        s = "(...)"
    return f"{name}{s}"


def api_body():
    items, toc = [], []
    for name in ssmix.__all__:
        obj = getattr(ssmix, name)
        toc.append(f'<li><a href="#{name}"><code>{name}</code></a></li>')
        kind = "class" if inspect.isclass(obj) else "function"
        doc = inspect.getdoc(obj) or ""
        block = [f'<div class="api-item" id="{name}"><h3>{kind} {name}</h3>',
                 f'<p class="sig">{html.escape(sig(obj, "ssmix." + name))}</p>',
                 f'<pre class="doc">{html.escape(doc)}</pre>']
        if inspect.isclass(obj):
            for mname, f in vars(obj).items():
                if mname.startswith("_") or not callable(f) or mname == "extra_repr":
                    continue
                mdoc = inspect.getdoc(f) or ""
                block.append(f'<div class="method"><p class="sig"><b>{html.escape(mname)}</b>'
                             f'{html.escape(sig(f, "")[0:])}</p><pre class="doc">{html.escape(mdoc)}</pre></div>')
        block.append("</div>")
        items.append("\n".join(block))
    return (f"<h1>API reference</h1>\n<p class=lead>Everything below is importable from <code>ssmix</code>. "
            f"All modules are parameter-free <code>torch.nn.Module</code>s that run on CPU or CUDA and take "
            f"<code>(batch, tokens, channels)</code> unless stated.</p>\n<ul class=toc>{''.join(toc)}</ul>\n"
            + "\n".join(items))


# -------------------------------------------------------------------- content
R = {}  # benchmark numbers, filled from benchmarks/results.json (see main)

NAMES = {
    "ssmix": ("ssmix AudioEmbedder", "none"),
    "handcrafted_all": ("All eight hand-crafted sets, concatenated", "none"),
    "compare": ("openSMILE ComParE 2016 (6,373 functionals)", "none"),
    "egemaps": ("openSMILE eGeMAPS v02 (88 functionals)", "none"),
    "librosa": ("librosa ESC bundle (MFCC, Δ, chroma, contrast, tonnetz, spectral)", "none"),
    "scattering": ("Wavelet scattering, orders 0–2 (kymatio, J=9, Q=8)", "none"),
    "gfcc": ("Gammatone cepstral coefficients + Δ", "none"),
    "mfcc": ("MFCC + Δ, mean/std", "none"),
    "logmel": ("Log-mel, mean/std", "none"),
    "cqt": ("Constant-Q spectrum, mean/std", "none"),
    "random_transformer": ("Random-init Transformer (4 layers, d=256)", "none"),
    "vggish": ("VGGish", "YouTube-8M"),
    "wav2vec2": ("wav2vec 2.0 base (all layers, mean-pooled)", "LibriSpeech 960 h"),
    "passt": ("PaSST-S", "AudioSet"),
    "efficientat_mn10": ("EfficientAT MobileNet mn10", "AudioSet"),
    "panns": ("PANNs CNN14", "AudioSet"),
}


def esc(n):
    return R["esc50"][n]


def wins(a, b):
    return sum(x > y for x, y in zip(esc(a)["folds"], esc(b)["folds"]))


def esc_rows():
    order = sorted(R["esc50"], key=lambda n: -esc(n)["mean"])
    return "".join(f'<tr{" class=ours" if n == "ssmix" else ""}><td>{NAMES[n][0]}</td><td>{NAMES[n][1]}</td>'
                   f'<td class=num>{esc(n)["mean"]} ± {esc(n)["std"]}</td></tr>' for n in order)


def fold_rows(others):
    rows = [("ssmix", " class=ours")] + [(o, "") for o in others]
    return "".join(f'<tr{c}><td>{NAMES[n][0].split(" (")[0]}</td>' + "".join(f"<td class=num>{v}</td>" for v in esc(n)["folds"])
                   + (f"<td class=num>{wins('ssmix', n)}/5</td>" if n != "ssmix" else "<td class=num>—</td>") + "</tr>"
                   for n, c in rows)


def fsdd_rows():
    order = sorted(R["fsdd"], key=lambda n: -R["fsdd"][n])
    return "".join(f'<tr{" class=ours" if n == "ssmix" else ""}><td>{NAMES[n][0]}</td><td>{NAMES[n][1]}</td>'
                   f'<td class=num>{R["fsdd"][n]}</td></tr>' for n in order)


def speed_rows():
    return "".join(f"<tr><td class=num>{T:,}</td><td class=num>{a}</td><td class=num>{b}</td></tr>"
                   for T, a, b, c in R["speed"]["rows"])


def speedup16k():
    r = next(r for r in R["speed"]["rows"] if r[0] == 16384)
    return round(r[3] / r[2], 1), r[2], r[3]


def index_body():
    return f"""
<h1>Training-free scale-space transformers on Daubechies wavelets</h1>
<p class=lead>ssmix replaces learned attention with exact multiscale operators: Daubechies wavelets of any
order, a GPU-batched tight frame, and a cross-scale attention mixer in which every wavelet coefficient
attends to its siblings, parent and children. Nothing is trained.</p>

<div class=cards>
<div class=card><b>ESC-50, linear probe</b><span class=big>{esc('ssmix')['mean']}%</span><br><span class=note>no training · ahead of VGGish, wav2vec 2.0 and openSMILE ComParE</span></div>
<div class=card><b>Exact by construction</b><span class=big>10<sup>−12</sup></span><br><span class=note>reconstruction error (float64); filters identical to PyWavelets db1–db38</span></div>
<div class=card><b>Zero parameters</b><span class=big>0</span><br><span class=note>no weights, no data, no fine-tuning</span></div>
</div>

<h2>Install</h2>
<pre><code>pip install git+{REPO}</code></pre>
<p>Try it in the browser: <a href="https://huggingface.co/spaces/Angshul/ssmix-audio-embedder">live demo on Hugging Face Spaces</a>.</p>
<p>Requires Python ≥ 3.9, PyTorch ≥ 2.0, torchaudio, NumPy and mpmath.</p>

<h2>Embed audio in three lines</h2>
<pre><code>from ssmix import AudioEmbedder

emb = AudioEmbedder()                                  # no weights to download
E = emb.embed(waves, sample_rate=44100, length=5.0)    # list of 1-D arrays -> (N, D) numpy</code></pre>
<p>Inputs are resampled to 32 kHz. With <code>length</code> every clip is padded or cut to the same
duration, so all embeddings share one dimension. Use the embeddings with any linear model, k-NN or clustering.</p>

<h2>Use the mixers as layers</h2>
<pre><code>import torch
from ssmix import SWTCrossScaleMixer, ScaleSpaceMixer

x = torch.randn(8, 4096, 64)                           # (batch, tokens, channels)
mix = SWTCrossScaleMixer(order=4, heads=8, window=2, tau=0.1)
y = mix(x)                                             # same shape, cross-scale attention

frame = ScaleSpaceMixer(order=8, rule="identity")      # db8 tight frame
W = frame.analysis(x)                                  # (8, J+1, 4096, 64) scale-space coefficients
assert torch.allclose(frame(x), x, atol=1e-5)          # exact round trip</code></pre>

<h2>What is inside</h2>
<table>
<tr><th>Object</th><th>Role</th></tr>
<tr><td><a href="api.html#AudioEmbedder"><code>AudioEmbedder</code></a></td><td>Training-free audio transformer: log-mel → cross-scale mixer blocks → pooled scale-space readout.</td></tr>
<tr><td><a href="api.html#SWTCrossScaleMixer"><code>SWTCrossScaleMixer</code></a></td><td>Cross-scale attention on the undecimated, delay-aligned Daubechies frame; shift-equivariant.</td></tr>
<tr><td><a href="api.html#CrossScaleMixer"><code>CrossScaleMixer</code></a></td><td>The same attention on the decimated orthonormal wavelet tree.</td></tr>
<tr><td><a href="api.html#ScaleSpaceMixer"><code>ScaleSpaceMixer</code></a></td><td>FFT-batched Daubechies tight frame for any length, with parameter-free gate rules.</td></tr>
<tr><td><a href="api.html#DWT"><code>DWT</code></a></td><td>Periodic orthonormal DWT/IDWT, identical to PyWavelets <code>periodization</code>.</td></tr>
<tr><td><a href="api.html#daubechies"><code>daubechies</code></a></td><td>Daubechies filters of any order, built from scratch by spectral factorisation.</td></tr>
</table>
<p>The full write-up is the <a href="article.html">article</a>; the construction is on <a href="method.html">Method</a>
and the measurements on <a href="benchmarks.html">Benchmarks</a>.</p>
"""


def method_body():
    return r"""
<h1>Method</h1>
<p class=lead>Three exact building blocks: Daubechies filters of controllable support, a tight frame
that computes every scale in one batched FFT, and a parameter-free cross-scale attention defined on that frame.</p>

<h2>1. Daubechies filters of any order</h2>
<p>Order \(N\) gives \(N\) vanishing moments and support \(2N-1\). The lowpass filter is obtained from
\(|H(\omega)|^2 = 2\cos^{2N}(\omega/2)\,P(\sin^2(\omega/2))\) with
\(P(y)=\sum_{k\lt N}\binom{N-1+k}{k}y^k\). Each root of \(P\) maps to a pair \(z, 1/z\); keeping
\(|z|\lt 1\) gives the extremal-phase factor. Roots are found at high precision with mpmath, so large
orders remain exact. The highpass is \(g[n] = (-1)^n h[L-1-n]\).</p>

<h2>2. A tight frame in one FFT</h2>
<p>The undecimated transform of a length-\(T\) signal has the band transfer functions</p>
<p>\[ D_j(\omega)=\tilde G(2^{j-1}\omega)\prod_{k\lt j-1}\tilde H(2^k\omega),\qquad
A_J(\omega)=\prod_{k\lt J}\tilde H(2^k\omega),\qquad \tilde H = H/\sqrt2,\ \tilde G = G/\sqrt2 . \]</p>
<p>Because \(|\tilde H|^2+|\tilde G|^2=1\), the sum telescopes to \(\sum_s|\Psi_s(\omega)|^2=1\): a
Parseval tight frame, with exact reconstruction by the adjoint for any \(T\). The responses are evaluated
on the DFT grid, so all \(J+1\) bands come from one rFFT and one batched inverse. The cost,
\(O(T\log T)\) per band, does not depend on the filter support. Each band receives an integer
group-delay compensation (a pure phase), so index \(t\) refers to the same instant at every scale.</p>

<h2>3. Cross-scale attention</h2>
<p>Tokens are the frame coefficients \(w_s(t)\in\mathbb R^C\). Token \((s,t)\) attends to the
neighbourhood</p>
<p>\[ \mathcal N(s,t)=\{(s',\,t+o\,2^{\ell(s')}) : s'\in\{s-1,s,s+1\},\ |o|\le k\}, \]</p>
<p>that is, its dyadic siblings, parent and children on the undecimated grid (\(\ell\) is the dyadic level of
a band). With per-head cosine weights</p>
<p>\[ a_{n}=\operatorname{softmax}_{n\in\mathcal N(s,t)}\frac{\langle\hat w_s(t),\hat w_n\rangle}{\tau},\qquad
w'_s(t)=\sum_{n\in\mathcal N(s,t)}a_n\,w_n , \]</p>
<p>followed by the exact synthesis. The operator has no parameters. It is shift-equivariant and
1-homogeneous, and it is exactly the identity as \(\tau\to0\), because each token's cosine with itself is maximal.
Each head's channels are contiguous and keys are never materialised. Memory is therefore
\(O(SCT)\) and the work is \(O(SCT\,(2k+1)\cdot 3)\), linear in sequence length.</p>

<h2>4. Audio embedder</h2>
<p>The <code>AudioEmbedder</code> runs at 32 kHz with 128 mel bands (25 ms window, 10 ms hop) and per-clip
normalisation, followed by two cross-scale mixer blocks (db4, 16 heads, \(k=2\), \(\tau=0.1\)). For each block the readout
pools over time:</p>
<ul>
<li>band magnitudes \(\log\overline{|w_s|}\) before and after mixing;</li>
<li>the mean attention weight of every cross-scale link and head, which records how strongly each
scale couples to its parent, children and siblings;</li>
<li>a spectral scale space along the mel axis, and the state statistics.</li>
</ul>
<p>The mean log-mel envelope is appended once, at the end.</p>
"""


def benchmarks_body():
    sp, ours16, att16 = speedup16k()
    S = R["speed"]
    return f"""
<h1>Benchmarks</h1>
<p class=lead>Frozen embeddings and one linear probe for every model. ssmix is never trained; the probe is the only
fitted component.</p>

<h2>Protocol</h2>
<ul>
<li><b>Probe:</b> StandardScaler, then multinomial logistic regression. C is chosen from
{{10<sup>−3</sup>, 10<sup>−2</sup>, 10<sup>−1</sup>, 1}} by 3-fold cross-validation inside each training split.
The probe is identical for every model.</li>
<li><b>ESC-50:</b> 2,000 five-second clips in 50 environmental classes, official 5-fold cross-validation.</li>
<li><b>FSDD:</b> Free Spoken Digit Dataset, 3,000 recordings of the digits 0–9, official split (takes 0–4 form the test set).
Clips are zero-padded to 2.3 s; PANNs CNN14 receives each clip at its true length.</li>
<li>The ssmix configuration was selected on ESC-50 folds 1–4 only; fold 5 never influenced a design choice.</li>
<li>Pretrained baselines use the authors' released weights; wav2vec 2.0 uses the mean-pooled states of all 13 layers.</li>
</ul>
<p>Reproduce with the scripts in <a href="{REPO}/tree/main/benchmarks"><code>benchmarks/</code></a>.</p>

<h2>ESC-50 (accuracy %, 5-fold mean ± std)</h2>
<table><tr><th>Model</th><th>Pretraining</th><th class=num>Accuracy</th></tr>{esc_rows()}</table>
<h3>Per fold</h3>
<table><tr><th>Model</th><th class=num>Fold 1</th><th class=num>Fold 2</th><th class=num>Fold 3</th><th class=num>Fold 4</th>
<th class=num>Fold 5</th><th class=num>Folds won by ssmix</th></tr>{fold_rows(["vggish", "wav2vec2", "compare"])}</table>

<h2>FSDD (accuracy %, official split)</h2>
<table><tr><th>Model</th><th>Pretraining</th><th class=num>Accuracy</th></tr>{fsdd_rows()}</table>

<h2>Speed</h2>
<p>CPU with 2 threads, batch 1, 64 channels, fp32; times in milliseconds.</p>
<table><tr><th class=num>Tokens T</th><th class=num>Tight frame (db4, J = 8)</th><th class=num>Cross-scale mixer (db4, J = 8, k = 2)</th></tr>{speed_rows()}</table>
<ul>
<li>At 16,384 tokens the cross-scale mixer is {sp}× faster than parameter-free softmax attention
({ours16} ms vs {att16:,} ms). Its cost grows near-linearly in T, while attention grows quadratically.</li>
<li>The tight frame's cost is flat in support: {S['db1']} ms with db1, {S['db4']} ms with db4 and
{S['db16']} ms with db16 at 16,384 tokens.</li>
<li>Embedding all 2,000 ESC-50 clips (10,000 s of audio) takes {S['esc50_seconds']} s on the same CPU.</li>
</ul>
<p>Run <code>python benchmarks/speed.py --device cuda</code> for GPU timings.</p>
"""


ARTICLE = r"""
<article class=paper>
<h1>ssmix: Training-Free Scale-Space Transformers for Audio on Daubechies Wavelets</h1>
<p class=authors>Angshul Majumdar · Indraprastha Institute of Information Technology Delhi</p>

<div class=abstract><b>Abstract.</b>
We present ssmix, an audio transformer with no learnable parameters. Its token mixer acts on an exact
multiscale representation: an undecimated Daubechies frame whose filter support is a free parameter.
All scales are computed in one batched FFT; the frame is Parseval-tight, so reconstruction is exact.
On this frame, every coefficient attends to its dyadic siblings, parent and children through
parameter-free cosine attention. The resulting embedder needs no data, no pretraining and no fine-tuning.
With a linear probe on frozen embeddings, it reaches @@ESC@@% on ESC-50. That is ahead of VGGish
(@@ESC_VGG@@%), wav2vec 2.0 (@@ESC_W2V@@%), openSMILE ComParE (@@ESC_CMP@@%) and the concatenation of eight
established hand-crafted feature sets (@@ESC_ALL@@%). On spoken digits (FSDD) it reaches @@FSDD@@%, ahead
of PaSST (@@F_PASST@@%), EfficientAT (@@F_EAT@@%), wav2vec 2.0 (@@F_W2V@@%) and PANNs (@@F_PANNS@@%). Because nothing is learned,
ssmix is intended for low-resource settings, where pretrained models have no matching data.</div>

<h2>1. Introduction</h2>
<p>Transformers learn how tokens mix. That learning costs data, compute and, for audio, large curated
corpora such as AudioSet or thousands of hours of speech. Many settings have none of these: under-documented
languages, specialised acoustic domains, small institutional collections. Hand-crafted features need no
data, but they are fixed summaries; they do not mix information across time and scale the way an attention
layer does.</p>
<p>ssmix keeps the transformer's structure (tokens, a token mixer, a readout) but replaces every learned
component with an exact operator. Wavelets are the natural basis: they separate a signal into scales with
compact, controllable support, and orthonormal Daubechies families give perfect reconstruction. The mixer
is attention over the wavelet tree: each coefficient looks at its neighbours in scale and time.</p>
<p><b>Contributions.</b></p>
<ol>
<li>A GPU-batched Daubechies tight frame for any order and any sequence length. All scales come from one FFT,
the cost does not depend on filter support, and reconstruction is exact.</li>
<li>A parameter-free cross-scale attention mixer on that frame. It is shift-equivariant, 1-homogeneous, exactly
the identity in its zero-temperature limit, and linear in sequence length.</li>
<li>A training-free audio embedder whose readout includes the attention maps themselves. Under an identical
linear probe it is ahead of VGGish, wav2vec 2.0, openSMILE ComParE and eight established
hand-crafted feature sets on ESC-50.</li>
</ol>

<h2>2. Related work</h2>
<p>The wavelet scattering transform [1, 2] is the classic training-free deep network: cascades of wavelet
transforms and modulus non-linearities. ssmix differs by mixing coefficients through data-dependent
attention across scales, rather than by fixed cascades. FNet [3] and WaveMix [4] replace attention with
Fourier and wavelet operators, but they keep learned layers and are trained end to end. Hand-crafted
audio descriptors, from MFCCs to the openSMILE eGeMAPS [5] and ComParE [6] sets, remain the standard
training-free baselines. Pretrained embedders, among them VGGish [7], wav2vec 2.0 [8], PANNs [9], PaSST [10] and
EfficientAT [11], are the standard learned ones.</p>

<h2>3. Method</h2>
<h3>3.1 Daubechies filters of any order</h3>
<p>Order \(N\) gives \(N\) vanishing moments and support \(2N-1\) [12]. The lowpass filter is obtained from
\(|H(\omega)|^2 = 2\cos^{2N}(\omega/2)\,P(\sin^2(\omega/2))\), with
\(P(y)=\sum_{k\lt N}\binom{N-1+k}{k}y^k\), by spectral factorisation. The roots of \(P\) are found at high
precision, and the minimum-phase factor is kept. The highpass is \(g[n]=(-1)^n h[L-1-n]\). The filters are
identical to PyWavelets for db1–db38 (max deviation below \(10^{-12}\)). Orthonormality and vanishing
moments are verified to db60.</p>

<h3>3.2 A tight frame in one FFT</h3>
<p>For the undecimated transform of a length-\(T\) sequence, the band transfer functions are</p>
<p>\[ D_j(\omega)=\tilde G(2^{j-1}\omega)\prod_{k\lt j-1}\tilde H(2^k\omega),\qquad
A_J(\omega)=\prod_{k\lt J}\tilde H(2^k\omega),\qquad \tilde H=H/\sqrt2,\ \tilde G=G/\sqrt2. \]</p>
<p>Since \(|\tilde H|^2+|\tilde G|^2=1\), the sum telescopes to \(\sum_s|\Psi_s(\omega)|^2=1\), a Parseval tight
frame; the adjoint is the exact inverse for every \(T\). Evaluated on the DFT grid, all \(J+1\) bands come
from one forward FFT and one batched inverse, independent of the filter support. Each band is shifted by
its integer group delay (a pure phase), so index \(t\) denotes the same instant at every scale. Translation-invariant
analysis of this kind goes back to [13, 14].</p>

<h3>3.3 Cross-scale attention</h3>
<p>Tokens are the coefficients \(w_s(t)\in\mathbb R^C\). Token \((s,t)\) attends to</p>
<p>\[ \mathcal N(s,t)=\{(s',\,t+o\,2^{\ell(s')}) : s'\in\{s-1,s,s+1\},\ |o|\le k\}, \]</p>
<p>its siblings, parent and children on the undecimated dyadic grid. With per-head cosine scores,</p>
<p>\[ a_n=\operatorname{softmax}_{n\in\mathcal N(s,t)}\frac{\langle\hat w_s(t),\hat w_n\rangle}{\tau},\qquad
w'_s(t)=\sum_{n\in\mathcal N(s,t)}a_n\,w_n , \]</p>
<p>followed by exact synthesis. The operator has no parameters. It is shift-equivariant and 1-homogeneous. As
\(\tau\to0\) it is exactly the identity, because a token's cosine with itself is maximal. Memory is \(O(SCT)\) and work
\(O(SCT(2k+1)\cdot3)\), linear in \(T\). Each head's channels are kept contiguous, and keys are never materialised.</p>

<h3>3.4 The embedder</h3>
<p>Audio at 32 kHz becomes a 128-band log-mel sequence (25 ms windows, 10 ms hop), normalised per clip. Two
blocks of cross-scale attention follow (db4, 16 heads, \(k=2\), \(\tau=0.1\)). For each block, the readout pools over time:</p>
<ul>
<li>band magnitudes before and after mixing;</li>
<li>the mean attention weight of every cross-scale link and head, a direct measurement of how each scale couples to
its parent, children and siblings;</li>
<li>a spectral Daubechies scale space along the mel axis, and the state statistics.</li>
</ul>
<p>The mean log-mel envelope is appended. The configuration was chosen on ESC-50 folds 1–4 and spoken-digit
development speakers only.</p>

<h2>4. Experiments</h2>
<h3>4.1 Protocol</h3>
<p>Every model produces a frozen clip embedding. One probe is used throughout: standardisation followed by
multinomial logistic regression, with C chosen from \(\{10^{-3},10^{-2},10^{-1},1\}\) by 3-fold cross-validation
inside each training split.</p>
<ul>
<li><b>ESC-50</b> [15]: 2,000 five-second clips in 50 classes, official 5-fold cross-validation.</li>
<li><b>FSDD</b> [16]: 3,000 spoken digits, official split.</li>
</ul>
<p>Pretrained models use their released weights. wav2vec 2.0 is given the mean-pooled states of all 13 layers, the
strongest standard frozen readout under a linear probe. PANNs receives each digit at its true length.</p>

<h3>4.2 Environmental sound (ESC-50)</h3>
<table><tr><th>Model</th><th>Pretraining</th><th class=num>Accuracy (%)</th></tr>@@ESC_ROWS@@</table>
<p class=caption>Table 1. ESC-50, 5-fold mean ± std. Hand-crafted sets and ssmix use no training data.</p>
<table><tr><th>Model</th><th class=num>Fold 1</th><th class=num>Fold 2</th><th class=num>Fold 3</th><th class=num>Fold 4</th>
<th class=num>Fold 5</th><th class=num>Folds won by ssmix</th></tr>@@FOLD_ROWS@@</table>
<p class=caption>Table 2. Per-fold accuracy. Fold 5 was never used for design.</p>
<p>ssmix is ahead of VGGish on @@W_VGG@@ of 5 folds, of wav2vec 2.0 on @@W_W2V@@ and of openSMILE ComParE on @@W_CMP@@.
It is also ahead of the concatenation of all eight hand-crafted sets (@@ESC_ALL@@%), a 7,519-dimensional
representation, while using no training data and no learned weights. Fold 5 was held out from all
design choices; there ssmix leads VGGish by @@F5_VGG@@ points and wav2vec 2.0 by @@F5_W2V@@.</p>

<h3>4.3 Spoken digits (FSDD)</h3>
<table><tr><th>Model</th><th>Pretraining</th><th class=num>Accuracy (%)</th></tr>@@FSDD_ROWS@@</table>
<p class=caption>Table 3. FSDD, official split.</p>
<p>ssmix reaches @@FSDD@@%, ahead of the AudioSet-trained PaSST and EfficientAT, wav2vec 2.0 (pretrained on 960 hours of
speech) and PANNs, with no exposure to speech or to any labelled audio.</p>

<h3>4.4 Exactness and cost</h3>
<p>The frame reconstructs to \(10^{-12}\) in float64 for every tested length, prime lengths included. The decimated
transform matches PyWavelets' periodized DWT to machine precision. At 16,384 tokens the cross-scale mixer runs
@@SPEEDUP@@× faster than parameter-free softmax attention on CPU, and the frame's cost is flat in filter support
(@@DB1@@ ms for db1, @@DB16@@ ms for db16). Embedding all of ESC-50 (10,000 s of audio) takes @@ESC_SEC@@ s on two CPU threads.</p>

<h2>5. Conclusion</h2>
<p>A transformer's token mixer does not have to be learned. Exact wavelet analysis with controllable support,
together with attention defined on the wavelet tree, gives an audio embedder with zero parameters. Under an
identical linear probe, it is ahead of VGGish, wav2vec 2.0 and the established hand-crafted feature sets on
environmental sound. It needs no data at all, which makes it a natural starting point for low-resource corpora.</p>

<h2>Reproducibility</h2>
<p>Code, tests and benchmark scripts: <a href="@@REPO@@">@@REPO_SHORT@@</a>. Live demo on Hugging Face Spaces:
<a href="https://huggingface.co/spaces/Angshul/ssmix-audio-embedder">Angshul/ssmix-audio-embedder</a>.
Every number on this page is produced by <code>benchmarks/report.py</code>.</p>

<h2>References</h2>
<ol class=refs>
<li>S. Mallat. Group invariant scattering. <i>Communications on Pure and Applied Mathematics</i>, 65(10), 2012.</li>
<li>J. Andén and S. Mallat. Deep scattering spectrum. <i>IEEE Transactions on Signal Processing</i>, 62(16), 2014.</li>
<li>J. Lee-Thorp, J. Ainslie, I. Eckstein and S. Ontañón. FNet: Mixing tokens with Fourier transforms. <i>NAACL</i>, 2022.</li>
<li>P. Jeevan and A. Sethi. WaveMix: Resource-efficient token mixing for images. arXiv:2203.03689, 2022.</li>
<li>F. Eyben et al. The Geneva minimalistic acoustic parameter set (GeMAPS) for voice research and affective computing. <i>IEEE Transactions on Affective Computing</i>, 7(2), 2016.</li>
<li>B. Schuller et al. The INTERSPEECH 2016 computational paralinguistics challenge. <i>Interspeech</i>, 2016.</li>
<li>S. Hershey et al. CNN architectures for large-scale audio classification. <i>ICASSP</i>, 2017.</li>
<li>A. Baevski, H. Zhou, A. Mohamed and M. Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. <i>NeurIPS</i>, 2020.</li>
<li>Q. Kong et al. PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. <i>IEEE/ACM TASLP</i>, 28, 2020.</li>
<li>K. Koutini, J. Schlüter, H. Eghbal-zadeh and G. Widmer. Efficient training of audio transformers with patchout. <i>Interspeech</i>, 2022.</li>
<li>F. Schmid, K. Koutini and G. Widmer. Efficient large-scale audio tagging via transformer-to-CNN knowledge distillation. <i>ICASSP</i>, 2023.</li>
<li>I. Daubechies. Orthonormal bases of compactly supported wavelets. <i>Communications on Pure and Applied Mathematics</i>, 41(7), 1988.</li>
<li>R. R. Coifman and D. L. Donoho. Translation-invariant de-noising. In <i>Wavelets and Statistics</i>, Springer, 1995.</li>
<li>T. Lindeberg. <i>Scale-Space Theory in Computer Vision</i>. Springer, 1994.</li>
<li>K. J. Piczak. ESC: Dataset for environmental sound classification. <i>ACM Multimedia</i>, 2015.</li>
<li>Z. Jackson et al. Free Spoken Digit Dataset. GitHub: Jakobovski/free-spoken-digit-dataset, 2018.</li>
</ol>
</article>
"""


def article_body():
    sp, _, _ = speedup16k()
    E = R["esc50"]
    rep = {
        "@@ESC@@": E["ssmix"]["mean"], "@@ESC_VGG@@": E["vggish"]["mean"], "@@ESC_W2V@@": E["wav2vec2"]["mean"],
        "@@ESC_CMP@@": E["compare"]["mean"], "@@ESC_ALL@@": E["handcrafted_all"]["mean"],
        "@@FSDD@@": R["fsdd"]["ssmix"], "@@F_PASST@@": R["fsdd"]["passt"], "@@F_EAT@@": R["fsdd"]["efficientat_mn10"],
        "@@F_W2V@@": R["fsdd"]["wav2vec2"], "@@F_PANNS@@": R["fsdd"]["panns"],
        "@@ESC_ROWS@@": esc_rows(), "@@FOLD_ROWS@@": fold_rows(["vggish", "wav2vec2", "compare"]),
        "@@FSDD_ROWS@@": fsdd_rows(),
        "@@W_VGG@@": wins("ssmix", "vggish"), "@@W_W2V@@": wins("ssmix", "wav2vec2"), "@@W_CMP@@": wins("ssmix", "compare"),
        "@@F5_VGG@@": round(E["ssmix"]["folds"][4] - E["vggish"]["folds"][4], 1),
        "@@F5_W2V@@": round(E["ssmix"]["folds"][4] - E["wav2vec2"]["folds"][4], 1),
        "@@SPEEDUP@@": sp, "@@DB1@@": R["speed"]["db1"], "@@DB16@@": R["speed"]["db16"],
        "@@ESC_SEC@@": R["speed"]["esc50_seconds"], "@@REPO@@": REPO, "@@REPO_SHORT@@": REPO.replace("https://", ""),
    }
    out = ARTICLE
    for k, v in rep.items():
        out = out.replace(k, str(v))
    assert "@@" not in out
    return out


def main():
    import json
    with open(os.path.join(ROOT, "benchmarks", "results.json")) as f:
        R.update(json.load(f))
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "style.css"), "w") as f:
        f.write(CSS.lstrip())
    bodies = {"index.html": (index_body(), False), "article.html": (article_body(), True),
              "method.html": (method_body(), True), "api.html": (api_body(), False),
              "benchmarks.html": (benchmarks_body(), False)}
    for fname, title in PAGES:
        body, math = bodies[fname]
        with open(os.path.join(DOCS, fname), "w") as f:
            f.write(page(fname, title, body, math))
    print("wrote", ", ".join(["style.css"] + [p for p, _ in PAGES]))


if __name__ == "__main__":
    main()
