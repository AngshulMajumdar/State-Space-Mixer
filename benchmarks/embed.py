"""Frozen embeddings on a prepared dataset.

    python benchmarks/embed.py <esc50|fsdd> <model> [<model> ...]

Models: ssmix, logmel, mfcc, scattering, random_transformer, egemaps, compare, librosa, gfcc, cqt,
handcrafted_all, vggish, efficientat_mn10, passt, wav2vec2, whisper, panns.
wav2vec2 / whisper / panns read their weights from weights/ (or SSMIX_WEIGHTS):
    weights/wav2vec2/{config.json, preprocessor_config.json, pytorch_model.bin}   facebook/wav2vec2-base
    weights/whisper/{config.json, preprocessor_config.json, model.safetensors}     openai/whisper-base
    weights/panns/Cnn14_mAP=0.431.pth                                              zenodo.org/records/3987831
and PANNs' model code (models.py, pytorch_utils.py from pip panns_inference) in ext/panns/.
The pretrained baselines need their code next to this repository (set SSMIX_EXT):
    git clone https://github.com/fschmid56/EfficientAT
    git clone https://github.com/harritaylor/torchvggish
and `pip install hear21passt` (weights are fetched from the authors' GitHub releases).
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.environ.get("SSMIX_EXT", os.path.join(ROOT, "ext"))
sys.path.insert(0, ROOT)
C = os.path.join(ROOT, "benchmarks", "cache")


def batched(fn, X, bs):
    out = []
    for i in range(0, len(X), bs):
        with torch.no_grad():
            out.append(fn(torch.from_numpy(X[i:i + bs])).float().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ ssmix
def m_ssmix(z):
    from ssmix import AudioEmbedder
    return batched(AudioEmbedder(), z["w32"], 16)


# ----------------------------------------------------- training-free baselines
def _logmel():
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=64)


def m_logmel(z):
    mel = _logmel()

    def f(x):
        X = torch.log(mel(x) + 1e-6)
        return torch.cat([X.mean(-1), X.std(-1)], 1)
    return batched(f, z["w16"], 64)


def m_mfcc(z):
    import torchaudio
    mf = torchaudio.transforms.MFCC(16000, n_mfcc=40, melkwargs=dict(n_fft=400, hop_length=160, n_mels=64))

    def f(x):
        M = mf(x)
        d = torchaudio.functional.compute_deltas(M)
        return torch.cat([M.mean(-1), M.std(-1), d.mean(-1), d.std(-1)], 1)
    return batched(f, z["w16"], 64)


def m_scattering(z):
    from kymatio.scattering1d.frontend.torch_frontend import ScatteringTorch1D
    S = ScatteringTorch1D(J=9, shape=z["w16"].shape[1], Q=8)  # 32 ms averaging, orders 0-2
    return batched(lambda x: torch.log(S(x).clamp_min(0) + 1e-6).mean(-1), z["w16"], 16)


def m_random_transformer(z):
    import torch.nn as nn
    torch.manual_seed(0)
    mel, d = _logmel(), 256
    proj = nn.Linear(64, d)
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 1024, dropout=0.0, batch_first=True,
                                                           norm_first=True), 4).eval()

    def f(x):
        X = torch.log(mel(x) + 1e-6).transpose(1, 2)
        X = X - X.mean(1, keepdim=True)
        T = X.shape[1]
        pos = torch.arange(T)[:, None] / (10000 ** (torch.arange(0, d, 2)[None] / d))
        pe = torch.zeros(T, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos), torch.cos(pos)
        H = enc(proj(X) + pe)
        return torch.cat([H.mean(1), H.std(1)], 1)
    return batched(f, z["w16"], 32)


# ------------------------------------------------- established hand-crafted sets
def _per_clip(fn, X):
    return np.stack([fn(x) for x in X])


def m_egemaps(z):  # openSMILE eGeMAPS v02 functionals (88)
    import opensmile
    s = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    return _per_clip(lambda x: s.process_signal(x, 16000).values[0], z["w16"])


def m_compare(z):  # openSMILE ComParE 2016 functionals (6373)
    import opensmile
    s = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016, feature_level=opensmile.FeatureLevel.Functionals)
    return _per_clip(lambda x: s.process_signal(x, 16000).values[0], z["w16"])


def _stats(M):
    return np.concatenate([M.mean(1), M.std(1)])


def m_librosa(z):  # the classic ESC feature bundle, mean/std over frames
    import librosa

    def f(x):
        x = x.astype(np.float32)
        S = np.abs(librosa.stft(x, n_fft=512, hop_length=160))
        mf = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=40, n_fft=512, hop_length=160)
        parts = [mf, librosa.feature.delta(mf),
                 librosa.feature.chroma_stft(S=S ** 2, sr=16000),
                 librosa.feature.spectral_contrast(S=S, sr=16000),
                 librosa.feature.tonnetz(y=x, sr=16000),
                 librosa.feature.spectral_centroid(S=S, sr=16000), librosa.feature.spectral_bandwidth(S=S, sr=16000),
                 librosa.feature.spectral_rolloff(S=S, sr=16000), librosa.feature.spectral_flatness(S=S),
                 librosa.feature.zero_crossing_rate(x, frame_length=512, hop_length=160), librosa.feature.rms(S=S, frame_length=512)]
        T = min(p.shape[1] for p in parts)
        return _stats(np.vstack([p[:, :T] for p in parts]))
    return _per_clip(f, z["w16"])


def m_gfcc(z):  # gammatone cepstral coefficients + deltas, mean/std
    import librosa
    from spafe.features.gfcc import gfcc

    def f(x):
        G = gfcc(x.astype(np.float64) + 1e-9 * np.random.default_rng(0).standard_normal(len(x)), fs=16000,
                 num_ceps=20, nfilts=64).T
        return _stats(np.vstack([G, librosa.feature.delta(G)]))
    return _per_clip(f, z["w16"])


def m_cqt(z):  # log constant-Q spectrum (7 octaves x 12 bins), mean/std
    import librosa
    return _per_clip(lambda x: _stats(np.log(np.abs(librosa.cqt(x.astype(np.float32), sr=16000, hop_length=256,
                                                                    n_bins=84)) + 1e-6)), z["w16"])


# --------------------------------------------------------- pretrained baselines
def m_passt(z):
    from hear21passt.base import load_model, get_scene_embeddings
    model = load_model(mode="embed_only").eval()
    return batched(lambda x: get_scene_embeddings(x, model), z["w32"], 8)


def m_efficientat(z):
    sys.path.insert(0, os.path.join(EXT, "EfficientAT"))
    cwd = os.getcwd()
    os.chdir(os.path.join(EXT, "EfficientAT"))
    from models.mn.model import get_model
    from models.preprocess import AugmentMelSTFT
    from helpers.utils import NAME_TO_WIDTH
    model = get_model(width_mult=NAME_TO_WIDTH("mn10_as"), pretrained_name="mn10_as").eval()
    os.chdir(cwd)
    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320).eval()
    return batched(lambda x: model(mel(x).unsqueeze(1))[1], z["w32"], 16)


def m_vggish(z):
    sys.path.insert(0, os.path.join(EXT, "torchvggish"))
    from torchvggish.vggish import VGGish
    from hubconf import model_urls
    model = VGGish(urls=model_urls, postprocess=False, device=torch.device("cpu")).eval()
    out = []
    for w in z["w16"]:
        with torch.no_grad():
            e = model(w.astype(np.float64), 16000)
        out.append(torch.cat([e.mean(0), e.std(0)]).numpy())
    return np.stack(out)


WEIGHTS = os.environ.get("SSMIX_WEIGHTS", os.path.join(ROOT, "weights"))


def m_wav2vec2(z):
    """facebook/wav2vec2-base; mean over time of every hidden layer (13 x 768), concatenated."""
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    path = os.path.join(WEIGHTS, "wav2vec2")
    fe, model = Wav2Vec2FeatureExtractor.from_pretrained(path), Wav2Vec2Model.from_pretrained(path).eval()

    def f(x):
        inp = fe(list(x.numpy()), sampling_rate=16000, return_tensors="pt")
        hs = model(inp.input_values, output_hidden_states=True).hidden_states
        return torch.cat([h.mean(1) for h in hs], 1)
    return batched(f, z["w16"], 8)


def m_whisper(z):
    """openai/whisper-base encoder; mean over the frames covering the clip, every layer (7 x 512)."""
    from transformers import WhisperFeatureExtractor, WhisperModel
    path = os.path.join(WEIGHTS, "whisper")
    fe, enc = WhisperFeatureExtractor.from_pretrained(path), WhisperModel.from_pretrained(path).encoder.eval()
    n_valid = int(np.ceil(z["w16"].shape[1] / 320))  # encoder frames are 20 ms

    def f(x):
        feats = fe(list(x.numpy()), sampling_rate=16000, return_tensors="pt").input_features
        hs = enc(feats, output_hidden_states=True).hidden_states
        return torch.cat([h[:, :n_valid].mean(1) for h in hs], 1)
    return batched(f, z["w16"], 8)


def m_panns(z):
    """PANNs CNN14 (AudioSet, mAP 0.431); its 2048-d clip embedding at 32 kHz.
    Each clip is embedded at its true length (trailing zero padding removed, at least 0.5 s):
    CNN14's pooled embedding degrades strongly on zero-padded short clips."""
    sys.path.insert(0, os.path.join(EXT, "panns"))  # models.py + pytorch_utils.py from panns_inference
    import models as panns_models
    m = panns_models.Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                           classes_num=527)
    ck = torch.load(os.path.join(WEIGHTS, "panns", "Cnn14_mAP=0.431.pth"), map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    out = []
    with torch.no_grad():
        for w in z["w32"]:
            nz = np.nonzero(np.abs(w) > 0)[0]
            n = max(nz[-1] + 1 if len(nz) else 0, 16000)
            out.append(m(torch.from_numpy(w[:n]).unsqueeze(0))["embedding"][0].numpy())
    return np.stack(out)


HANDCRAFTED = ("mfcc", "logmel", "scattering", "egemaps", "compare", "librosa", "gfcc", "cqt")


def handcrafted_all(ds):
    """Concatenation of all eight hand-crafted sets (extract them first)."""
    def run(z):
        return np.nan_to_num(np.concatenate([np.load(f"{C}/emb_{ds}_{n}.npy") for n in HANDCRAFTED], 1))
    return run


MODELS = {
    "ssmix": m_ssmix, "logmel": m_logmel, "mfcc": m_mfcc, "scattering": m_scattering,
    "random_transformer": m_random_transformer, "vggish": m_vggish,
    "efficientat_mn10": m_efficientat, "passt": m_passt,
    "egemaps": m_egemaps, "compare": m_compare, "librosa": m_librosa, "gfcc": m_gfcc, "cqt": m_cqt,
    "wav2vec2": m_wav2vec2, "whisper": m_whisper, "panns": m_panns,
}

if __name__ == "__main__":
    ds = sys.argv[1]
    z = dict(np.load(f"{C}/{ds}_wav.npz"))
    MODELS["handcrafted_all"] = handcrafted_all(ds)
    for name in sys.argv[2:]:
        path = f"{C}/emb_{ds}_{name}.npy"
        if os.path.exists(path):
            print(name, "cached")
            continue
        t = time.time()
        E = MODELS[name](z)
        np.save(path, E.astype(np.float32))
        print(f"{ds:6s} {name:20s} dim={E.shape[1]:5d} {time.time() - t:7.1f}s", flush=True)
