"""ssmix: training-free scale-space transformers on Daubechies wavelets."""
from .crossscale import DWT, CrossScaleMixer, SWTCrossScaleMixer
from .daubechies import daubechies, daubechies_lowpass, qmf_highpass
from .embedder import AudioEmbedder
from .frame import ScaleSpaceMixer

__version__ = "0.1.0"

__all__ = [
    "AudioEmbedder",
    "SWTCrossScaleMixer",
    "CrossScaleMixer",
    "ScaleSpaceMixer",
    "DWT",
    "daubechies",
    "daubechies_lowpass",
    "qmf_highpass",
]
