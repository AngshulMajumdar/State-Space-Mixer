"""Cross-scale attention mixer on the decimated orthonormal Daubechies DWT.

Tokens are wavelet coefficients (s, t): band s = 1..J (details) plus the
approximation band A_J (treated as scale J+1, same resolution as D_J); band s
holds N / 2^s tokens, each a vector in R^C (C channels, split into H heads).

Neighbourhood of token (s, t):
    siblings  (s, t+o), |o| <= k             (circular)
    parent    (s+1, floor(t/2))   [D_J -> A_J at the same index]
    children  (s-1, 2t), (s-1, 2t+1)          [A_J -> D_J at the same index]

Weights are parameter-free cosine attention, per head:
    a = softmax_{n in N(s,t)} ( <w_hat(s,t), w_hat(n)> / tau ),  w' = sum_n a_n w(n)
One synchronous (Jacobi) update over all tokens, then the exact inverse DWT.
tau -> 0 gives w' = w (self has the maximal cosine), hence y = x exactly.

The transform is periodic, orthonormal (Parseval), computed with one batched
strided conv1d per level; synthesis with conv_transpose1d. Total tokens ~ T,
attention cost O(T C (2k+4)) -- linear in T.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .daubechies import daubechies


class DWT(nn.Module):
    """Periodic orthonormal DWT along the last axis of (..., N), N % 2^J == 0."""

    def __init__(self, order: int = 4):
        super().__init__()
        h, g = daubechies(order)
        self.L = len(h)
        # PyWavelets 'periodization':  cA[n] = sum_k h[k] x[(2n + k - (N-1)) mod len]  (same for cD with g)
        self.offset = self.L // 2 - 1
        self.register_buffer("w", torch.tensor([h, g], dtype=torch.float64).unsqueeze(1), persistent=False)

    def max_levels(self, N: int) -> int:
        """Largest J with N divisible by 2^J and the coarsest band no shorter than the filter."""
        J = 0
        while N % 2 == 0 and N // 2 >= max(self.L, 2):
            N //= 2
            J += 1
        return J

    def _w(self, x):
        return self.w.to(dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor, J: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """J-level DWT of (..., N). Returns ([D_1, ..., D_J], A_J), D_j of length N / 2^j.
        Identical to PyWavelets wavedec(x, f"db{order}", mode="periodization", level=J)."""
        shape = x.shape[:-1]
        a = x.reshape(-1, 1, x.shape[-1])
        w = self._w(x)
        ds = []
        for _ in range(J):
            n = a.shape[-1]
            if self.offset:
                a = torch.roll(a, shifts=self.offset % n, dims=-1)
            ap = torch.cat([a, a[..., : self.L - 1]], -1) if self.L > 1 else a
            if ap.shape[-1] < n + self.L - 1:  # filter longer than signal: wrap repeatedly
                reps = math.ceil((n + self.L - 1) / n)
                ap = a.repeat(1, 1, reps)[..., : n + self.L - 1]
            y = F.conv1d(ap, w, stride=2)
            ds.append(y[:, 1:2].reshape(*shape, -1))
            a = y[:, 0:1]
        return ds, a.reshape(*shape, -1)

    def inverse(self, ds: List[torch.Tensor], a: torch.Tensor) -> torch.Tensor:
        """Exact inverse (adjoint) of forward: ([D_1, ..., D_J], A_J) -> (..., N)."""
        shape = a.shape[:-1]
        w = self._w(a)
        a = a.reshape(-1, 1, a.shape[-1])
        for d in reversed(ds):
            n = a.shape[-1]
            y = F.conv_transpose1d(torch.cat([a, d.reshape(-1, 1, n)], 1), w, stride=2)  # (M, 1, 2n+L-2)
            N = 2 * n
            out = y[..., :N].clone()
            tail = y[..., N:]
            while tail.shape[-1] > 0:  # fold circular wrap-around
                m = min(N, tail.shape[-1])
                out[..., :m] += tail[..., :m]
                tail = tail[..., m:]
            a = torch.roll(out, shifts=-(self.offset % N), dims=-1) if self.offset else out
        return a.reshape(*shape, -1)


class CrossScaleMixer(nn.Module):
    """Parameter-free cross-scale attention token mixer. Input/output (B, T, C).

    Args:
        order:  Daubechies order N (support 2N-1).
        levels: J (None -> as many as the padded length allows, capped at max_levels).
        heads:  channel groups with independent attention.
        window: sibling half-width k.
        tau:    cosine-attention temperature (tau -> 0 is the identity).
        parent / children: include those links.
    """

    def __init__(self, order=4, levels: Optional[int] = None, max_levels=8, heads=1, window=2, tau=0.1,
                 parent=True, children=True, pad="reflect"):
        super().__init__()
        self.dwt = DWT(order)
        self.order, self.support = order, 2 * order - 1
        self.levels, self.max_levels = levels, max_levels
        self.heads, self.window, self.tau = heads, window, tau
        self.parent, self.children, self.pad = parent, children, pad
        self.last_attention = None

    # ---------------------------------------------------------------- layout
    def _levels(self, T):
        J = self.levels
        if J is None:
            J = max(1, min(self.max_levels, int(math.floor(math.log2(max(T, 2) / max(self.dwt.L, 2))))))
        return J

    def coefficients(self, x):
        """(B, T, C) -> ([D_1..D_J], A_J) each (B, C, N/2^s), plus padding info."""
        B, T, C = x.shape
        J = self._levels(T)
        N = math.ceil(T / 2 ** J) * 2 ** J
        xc = x.transpose(1, 2)
        if N > T:
            if self.pad == "reflect" and N - T < T:
                xc = F.pad(xc, (0, N - T), mode="reflect")
            else:
                xc = F.pad(xc, (0, N - T), mode="circular" if N - T <= T else "constant")
        ds, a = self.dwt(xc, J)
        return ds, a, T

    # -------------------------------------------------------------- attention
    def _attend(self, q, keys):
        """q: (B, n, C), keys: (B, n, M, C) -> mixed (B, n, C), weights (B, n, M, H)."""
        B, n, M, C = keys.shape
        H = self.heads
        D = C // H
        qh = q.view(B, n, 1, H, D)
        kh = keys.view(B, n, M, H, D)
        qn = qh / qh.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        kn = kh / kh.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        s = (qn * kn).sum(-1) / self.tau  # (B, n, M, H)
        a = torch.softmax(s, dim=2)
        out = (a.unsqueeze(-1) * kh).sum(2).reshape(B, n, C)
        return out, a

    def _siblings(self, b):  # b: (B, n, C) -> (B, n, 2k+1, C); offset 0 first
        k = self.window
        offs = [0] + [o for o in range(-k, k + 1) if o != 0]
        return torch.stack([torch.roll(b, shifts=-o, dims=1) for o in offs], 2)

    def mix(self, ds, a):
        """One synchronous cross-scale attention update. Bands: (B, C, n_s)."""
        bands = [d.transpose(1, 2) for d in ds] + [a.transpose(1, 2)]  # (B, n_s, C), index J = A_J
        J = len(ds)
        new, att = [], []
        for s, b in enumerate(bands):
            keys = [self._siblings(b)]
            if s < J - 1 and self.parent:  # parent at the next coarser detail band
                keys.append(bands[s + 1].repeat_interleave(2, dim=1)[:, : b.shape[1]].unsqueeze(2))
            elif s == J - 1 and self.parent:  # D_J -> A_J, same index
                keys.append(bands[J].unsqueeze(2))
            if self.children:
                if 0 < s < J:
                    c = bands[s - 1]
                    keys.append(c.view(c.shape[0], b.shape[1], 2, c.shape[2]))
                elif s == J:  # A_J -> D_J, same index
                    keys.append(bands[J - 1].unsqueeze(2))
            out, w = self._attend(b, torch.cat(keys, 2))
            new.append(out)
            att.append(w)
        self.last_attention = att
        nd = [x.transpose(1, 2) for x in new[:J]]
        return nd, new[J].transpose(1, 2)

    def forward(self, x):
        """(B, T, C) -> (B, T, C): DWT, one cross-scale attention update, inverse DWT."""
        ds, a, T = self.coefficients(x)
        ds, a = self.mix(ds, a)
        y = self.dwt.inverse(ds, a)  # (B, C, N)
        return y[..., :T].transpose(1, 2)


class SWTCrossScaleMixer(nn.Module):
    """Cross-scale attention on the undecimated, delay-aligned Daubechies tight frame.

    Every band s = 0..J (J details + approximation) has a token at every instant t.
    Token (s, t) attends to the bands s-1, s, s+1 at instants t + o * 2^{lev(s')},
    |o| <= k (lev = dyadic level of the key band), i.e. its dyadic siblings, parent
    and children on the same grid as the decimated tree, without decimation, so the
    operator is shift-equivariant. Weights: parameter-free cosine attention per head;
    values are the raw coefficients. tau -> 0 is the exact identity (self has
    cosine 1); synthesis is the tight-frame adjoint.  Input/output (B, T, C).
    """

    def __init__(self, order=4, levels=None, heads=8, window=1, tau=0.1, scales=(-1, 0, 1), pad="reflect"):
        super().__init__()
        from .frame import ScaleSpaceMixer
        self.frame = ScaleSpaceMixer(order=order, levels=levels, rule="identity", pad=pad)
        self.order, self.support = order, 2 * order - 1
        self.heads, self.window, self.tau, self.scales = heads, window, tau, tuple(scales)
        self.last_attention = None

    def coefficients(self, x):
        """(B, T, C) -> padded coefficients Wp (B, S, C, n), plan, T."""
        B, T, C = x.shape
        plan = self.frame._plan(T, x.device, x.dtype)
        Wp = self.frame._analysis_padded(self.frame._pad(x.transpose(1, 2), plan), plan)
        return Wp, plan, T

    def synthesize(self, Wp, plan, T):
        """Padded coefficients (B, S, C, n) -> (B, T, C) by the tight-frame adjoint (exact inverse)."""
        y = self.frame._synthesis_padded(Wp, plan)
        return self.frame._crop(y, plan, T).transpose(1, 2)

    def links(self, S):
        """Key list of every band: [(key band, time shift in samples)], self first."""
        J = S - 1
        lev = [s + 1 for s in range(J)] + [J]
        out = []
        for s in range(S):
            ks = [(s, 0)]
            for ds in self.scales:
                s2 = s + ds
                if not 0 <= s2 < S:
                    continue
                for o in range(-self.window, self.window + 1):
                    if ds == 0 and o == 0:
                        continue
                    ks.append((s2, o * 2 ** lev[s2]))
            out.append(ks)
        return out

    def mix(self, Wp):
        """One synchronous cross-scale attention update on padded coefficients (B, S, C, n).

        Layout (B, S, H, n, D) keeps each head's channels contiguous; tokens are normalised once
        and keys are never materialised, so memory is O(B S C n) and the work is
        O(B S C n (2k+1) |scales|) multiply-adds.
        """
        B, S, C, n = Wp.shape
        H = self.heads
        D = C // H
        Wt = Wp.reshape(B, S, H, D, n).transpose(3, 4).contiguous()  # (B, S, H, n, D)
        Wn = Wt / Wt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        out, att = [], []
        for s, ks in enumerate(self.links(S)):
            q = Wn[:, s]
            scores = torch.stack([(q * torch.roll(Wn[:, s2], -sh, dims=2)).sum(-1) if sh else (q * Wn[:, s2]).sum(-1)
                                  for s2, sh in ks], 1) / self.tau  # (B, M, H, n)
            a = torch.softmax(scores, dim=1)
            acc = torch.zeros_like(Wt[:, s])
            for m, (s2, sh) in enumerate(ks):
                v = torch.roll(Wt[:, s2], -sh, dims=2) if sh else Wt[:, s2]
                acc.addcmul_(a[:, m].unsqueeze(-1), v)
            out.append(acc)
            att.append(a)
        self.last_attention = att
        return torch.stack(out, 1).transpose(3, 4).reshape(B, S, C, n)

    def forward(self, x):
        """(B, T, C) -> (B, T, C): analysis, one cross-scale attention update, synthesis."""
        Wp, plan, T = self.coefficients(x)
        return self.synthesize(self.mix(Wp), plan, T)
