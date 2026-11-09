"""NX8 fake quantization in PyTorch: the IPU's 8-bit row format, vectorized, on CPU.

NX8 (ipu-emulator ADR 0007) is 128-element rows of at most 8-bit elements plus a
per-row ``e8m0`` scale: the largest FP32 exponent field of the row. Each element
stores its exponent as a *distance below* that scale; ``fe`` exponent bits and
``fm`` mantissa bits. This module reproduces ``ipu_isa.nx8.encode_row`` followed
by ``decode_row`` (value in, value out) with integer arithmetic on the FP32 bit
fields, so it is bit-exact with the emulator for static formats
(``tests/test_ipu_nx8.py`` checks it against ``ipu_isa`` when that is importable).

``dynamic`` rows choose fe = 1 or 2 with the spec's score (§4.3.2.3), evaluated
here in float64 instead of exact fractions: it can differ from the emulator on
a tie at the last bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

ROW = 128
_MAN = 23


@dataclass(frozen=True)
class Nx8:
    """One NX8 type: ``sign`` bits, ``fe`` exponent bits, ``bits`` total (or dynamic)."""

    fe: int = 4
    sign: int = 1
    bits: int = 8
    dynamic: bool = False

    @classmethod
    def parse(cls, name: str) -> Nx8:
        """``e4`` / ``e5`` / ``dynamic`` / ``s1e4w8`` (the ipu-harness variant names)."""
        name = name.lower().removeprefix("nx8_").removeprefix("nx8-")
        if name == "dynamic":
            return cls(fe=1, sign=1, bits=8, dynamic=True)
        if name.startswith("s") and "e" in name and "w" in name:
            s, rest = name[1:].split("e", 1)
            fe, w = rest.split("w", 1)
            return cls(fe=int(fe), sign=int(s), bits=int(w))
        if name.startswith("e"):
            return cls(fe=int(name[1:]))
        raise ValueError(f"unknown NX8 type {name!r}")

    def fm(self, fe: int | None = None) -> int:
        return self.bits - (self.fe if fe is None else fe) - self.sign


def _fields(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bits = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = bits >> 31
    exp = (bits >> _MAN) & 0xFF
    sig = (bits & 0x7FFFFF) | (1 << _MAN)
    return sign, exp, sig


def _rshift_rtn(v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """RTN(v / 2^s), an exact half rounding up; s may be <= 0."""
    s = s.clamp(max=62)
    pos = s > 0
    sp = torch.where(pos, s, torch.ones_like(s))
    up = (v + (torch.ones_like(v) << (sp - 1))) >> sp
    down = v << (-s).clamp(min=0, max=40)
    return torch.where(pos, up, down)


def _quant_rows(sign, exp, sig, scale, fe: int, fm: int, signed: int) -> torch.Tensor:
    """Decoded values of rows already split into fields; ``scale`` broadcasts per row."""
    clamp_exp = (1 << fe) - 1
    distance = scale - exp
    direct = distance < clamp_exp
    # direct branch
    man_d = _rshift_rtn(sig, torch.full_like(sig, _MAN - fm)) - (1 << fm)
    carry = man_d == (1 << fm)
    sat = carry & (distance <= 0)
    e_d = torch.where(carry & ~sat, distance - 1, torch.where(sat, torch.zeros_like(distance), distance))
    m_d = torch.where(carry & ~sat, torch.zeros_like(man_d), torch.where(sat, torch.full_like(man_d, (1 << fm) - 1), man_d))
    # clamp branch
    shift = distance - clamp_exp + 1
    man_c = _rshift_rtn(sig, _MAN - fm + shift)
    carry_c = man_c == (1 << fm)
    e_c = torch.where(carry_c, torch.full_like(distance, clamp_exp - 1), torch.full_like(distance, clamp_exp))
    m_c = torch.where(carry_c, torch.zeros_like(man_c), man_c)
    e = torch.where(direct, e_d, e_c)
    m = torch.where(direct, m_d, m_c)
    unb = scale - 127
    normal = e < clamp_exp
    frac = m.to(torch.float64) / float(1 << fm)
    mag = torch.where(normal, torch.ldexp(1.0 + frac, (unb - e).to(torch.float64)),
                      torch.ldexp(frac, (unb - (clamp_exp - 1)).to(torch.float64)))
    if signed:
        return torch.where(sign.bool(), -mag, mag)
    return torch.where(sign.bool(), torch.zeros_like(mag), mag)


def _dynamic_choice(exp, sig, scale) -> torch.Tensor:
    """Per row: 1 (E2M5) or 0 (E1M6), the §4.3.2.3 score in float64."""
    fm1 = 6
    distance = scale - exp
    mag = torch.ldexp(sig.to(torch.float64), (exp - scale - _MAN).to(torch.float64))
    step1 = torch.ldexp(torch.ones_like(mag), (-distance.clamp(max=1) - fm1).to(torch.float64))
    step2 = torch.ldexp(torch.ones_like(mag), (-distance.clamp(max=3) - fm1 + 1).to(torch.float64))
    err1 = mag - torch.floor(mag / step1) * step1
    err2 = mag - torch.floor(mag / step2) * step2
    is_a = distance == 0
    is_b = (distance > 1) & (distance <= fm1 + 2)
    w = torch.where(is_a, 2.0 ** (2 * fm1), 2.0 ** (2 * (fm1 + 1)))
    term = torch.floor((err2 * err2 - err1 * err1).abs() * w + 0.5)
    a = torch.where(is_a, term, torch.zeros_like(term)).sum(-1, keepdim=True)
    b = torch.where(is_b, term, torch.zeros_like(term)).sum(-1, keepdim=True)
    return (b - 4 * a >= 0).to(torch.int64)


def quantize(x: torch.Tensor, fmt: Nx8 | str = "e4", dim: int = -1) -> torch.Tensor:
    """``x`` rounded to NX8, rows of 128 along ``dim`` (the last row zero-padded)."""
    if isinstance(fmt, str):
        fmt = Nx8.parse(fmt)
    if x.numel() == 0:
        return x
    orig_dtype = x.dtype
    xt = x.float().movedim(dim, -1)
    shape = xt.shape
    n = shape[-1]
    pad = (-n) % ROW
    flat = xt.reshape(-1, n)
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    rows = flat.reshape(-1, ROW)
    finite = torch.isfinite(rows)
    rows = torch.where(finite, rows, torch.sign(rows) * torch.finfo(torch.float32).max)
    sign, exp, sig = _fields(rows)
    scale = exp.amax(-1, keepdim=True)
    # padding elements must not raise the scale (they are 0 → exp 0, harmless)
    if fmt.dynamic:
        choice = _dynamic_choice(exp, sig, scale)
        q1 = _quant_rows(sign, exp, sig, scale, 1, 6, 1)
        q2 = _quant_rows(sign, exp, sig, scale, 2, 5, 1)
        q = torch.where(choice.bool(), q2, q1)
    else:
        q = _quant_rows(sign, exp, sig, scale, fmt.fe, fmt.fm(), fmt.sign)
    q = q.reshape(flat.shape)[:, :n].reshape(shape)
    return q.movedim(-1, dim).to(orig_dtype)
