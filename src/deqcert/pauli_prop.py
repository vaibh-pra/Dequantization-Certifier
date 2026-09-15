"""Pauli-propagation backend.

Statevector simulation evolves 2^n amplitudes. This module works in the
Heisenberg picture instead: it propagates the readout observable backwards
through the circuit, expanded in the Pauli basis. Cost then scales with the
number of retained Pauli terms rather than with 2^n, which is what lifts the
certifier out of the ~12-qubit regime.

The point is not only speed. Because data enters this circuit family as
rotation angles, the data-dependent factors appear directly in the coefficients
as cos/sin of the encoding channels. Propagating symbolically therefore
*derives* the classical surrogate rather than fitting a guessed basis to circuit
evaluations, which is how :func:`deqcert.extract_surrogate` works.

Conventions
-----------
A Pauli string is the pair of bitmasks ``(xmask, zmask)`` representing the
Hermitian operator

    P(x, z) = i^{|x & z|} * X^x Z^z ,

so a qubit with both bits set carries Y. Bit ``q`` of a mask is qubit ``q``.

Conjugation by a Pauli rotation ``U = exp(-i phi G / 2)`` acts as

    U† P U = P                                   if [P, G] = 0
    U† P U = cos(phi) P + i sin(phi) G P         if {P, G} = 0

and ``i G P`` is again a Hermitian Pauli when G and P anticommute, so all
coefficients stay real.

Truncation
----------
A coefficient is a polynomial in cos/sin of the encoding channels. Since every
such factor is bounded by 1 in absolute value, the sum of the absolute values of
its monomial coefficients is a rigorous upper bound on the polynomial over all
inputs. Dropping a term therefore changes the final expectation by at most that
bound, and the accumulated bound is reported as ``truncation_bound`` -- a
guarantee, not an estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .certifier import EncoderSpec, _angle

# ---------------------------------------------------------------------------
# Pauli algebra on (xmask, zmask) bitmask pairs
# ---------------------------------------------------------------------------


def _popcount(v: int) -> int:
    return bin(v).count("1")


def pauli_mul(p1, p2):
    """Product of two Hermitian Pauli strings.

    Returns ``(p3, phase)`` with ``P1 P2 = phase * P3`` and phase in
    {1, i, -1, -i} represented as a complex number.
    """
    x1, z1 = p1
    x2, z2 = p2
    x3, z3 = x1 ^ x2, z1 ^ z2
    a = _popcount(x1 & z1) + _popcount(x2 & z2) - _popcount(x3 & z3)
    b = _popcount(z1 & x2)
    phase = (1j ** (a % 4)) * ((-1) ** (b % 2))
    return (x3, z3), phase


def commutes(p1, p2) -> bool:
    """Symplectic commutation test for two Pauli strings."""
    x1, z1 = p1
    x2, z2 = p2
    return (_popcount(x1 & z2) + _popcount(z1 & x2)) % 2 == 0


def pauli_weight(p) -> int:
    x, z = p
    return _popcount(x | z)


def pauli_str(p, n: int) -> str:
    x, z = p
    out = []
    for q in range(n):
        bx, bz = (x >> q) & 1, (z >> q) & 1
        out.append({(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[(bx, bz)])
    return "".join(out)


def single(letter: str, q: int):
    """Single-qubit Pauli on qubit ``q``."""
    if letter == "I":
        return (0, 0)
    if letter == "X":
        return (1 << q, 0)
    if letter == "Z":
        return (0, 1 << q)
    if letter == "Y":
        return (1 << q, 1 << q)
    raise ValueError(letter)


# ---------------------------------------------------------------------------
# symbolic coefficients: polynomials in cos/sin of encoding channels
# ---------------------------------------------------------------------------
# A monomial is a sorted tuple of (channel_key, 'c'|'s', power).
# A coefficient is {monomial: float}. The empty monomial () is the constant.

EMPTY: tuple = ()


def coef_scale(coef: dict, s: float) -> dict:
    if s == 0.0:
        return {}
    return {m: v * s for m, v in coef.items()}


def coef_add(a: dict, b: dict) -> dict:
    out = dict(a)
    for m, v in b.items():
        nv = out.get(m, 0.0) + v
        if nv == 0.0:
            out.pop(m, None)
        else:
            out[m] = nv
    return out


def coef_mul_trig(coef: dict, channel, kind: str) -> dict:
    """Multiply a coefficient by cos(channel) or sin(channel)."""
    out = {}
    for m, v in coef.items():
        d = {}
        for ch, k, p in m:
            d[(ch, k)] = d.get((ch, k), 0) + p
        d[(channel, kind)] = d.get((channel, kind), 0) + 1
        nm = tuple(sorted((ch, k, p) for (ch, k), p in d.items()))
        out[nm] = out.get(nm, 0.0) + v
    return out


def coef_bound(coef: dict) -> float:
    """Rigorous upper bound on |coefficient| over all inputs (|cos|,|sin| <= 1)."""
    return sum(abs(v) for v in coef.values())


def coef_eval(coef: dict, angles: dict) -> np.ndarray:
    """Evaluate given {channel_key: angle_array}."""
    total = None
    for m, v in coef.items():
        term = v
        for ch, k, p in m:
            f = np.cos(angles[ch]) if k == "c" else np.sin(angles[ch])
            term = term * (f ** p)
        total = term if total is None else total + term
    return 0.0 if total is None else total


# ---------------------------------------------------------------------------
# angle sources -> channels
# ---------------------------------------------------------------------------


def _src_key(src):
    """Hashable, normalised key for a data-carrying angle expression."""
    if src[0] == "data":
        return ("data", src[1], src[2])
    if src[0] == "data_poly":
        return ("data_poly", tuple((c, tuple(f)) for c, f in src[1]))
    raise ValueError(src)


def _halve_src(src):
    """The angle source for half the given angle (needed to split crot)."""
    if src[0] == "data":
        _, feat, scale = src
        if isinstance(scale, tuple):
            raise NotImplementedError(
                "controlled rotation with a trainable re-encoding scale is not "
                "supported by the symbolic backend; use the statevector path"
            )
        return ("data", feat, scale / 2.0)
    if src[0] == "data_poly":
        return ("data_poly", [(c / 2.0, f) for c, f in src[1]])
    raise ValueError(src)


def _is_data(src) -> bool:
    return src[0] in ("data", "data_poly")


# ---------------------------------------------------------------------------
# the propagator
# ---------------------------------------------------------------------------


@dataclass
class PropagationResult:
    """Surrogate for one readout, derived by Heisenberg propagation."""

    readout: str
    n_qubits: int
    terms: dict = field(default_factory=dict)   # monomial -> float
    n_pauli_terms: int = 0                      # Pauli strings alive at the end
    max_pauli_terms: int = 0                    # peak during propagation
    truncation_bound: float = 0.0               # rigorous error bound
    channels: list = field(default_factory=list)
    dropped: int = 0

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def evaluate(self, x, spec: EncoderSpec, params: dict) -> np.ndarray:
        """Evaluate the surrogate on inputs ``x`` -- no circuit involved."""
        x = np.atleast_2d(np.asarray(x, float))
        angles = {}
        for ch in self.channels:
            if ch[0] == "data":
                src = ("data", ch[1], ch[2])
            else:
                src = ("data_poly", [(c, list(f)) for c, f in ch[1]])
            angles[ch] = _angle(src, x, params)
        val = coef_eval(self.terms, angles)
        return np.broadcast_to(np.asarray(val, float), (x.shape[0],))


def _rot_conjugate(obs: dict, G, angle_src, params, half: bool = False):
    """Conjugate the observable by exp(-i phi G / 2) for every Pauli term."""
    if _is_data(angle_src):
        src = _halve_src(angle_src) if half else angle_src
        ch = _src_key(src)
        symbolic, cval, sval = True, None, None
    else:
        symbolic, ch = False, None
        phi = float(_angle(angle_src, np.zeros((1, 1)), params))
        if half:
            phi = phi / 2.0
        cval, sval = math.cos(phi), math.sin(phi)

    out = {}
    for P, coef in obs.items():
        if commutes(P, G):
            out[P] = coef_add(out.get(P, {}), coef)
            continue
        # cos(phi) * P
        if symbolic:
            kept = coef_mul_trig(coef, ch, "c")
        else:
            kept = coef_scale(coef, cval)
        if kept:
            out[P] = coef_add(out.get(P, {}), kept)
        # i sin(phi) * G P   (Hermitian because G and P anticommute)
        GP, phase = pauli_mul(G, P)
        w = (1j * phase).real
        if abs((1j * phase).imag) > 1e-12:
            raise AssertionError("non-real coefficient in anticommuting branch")
        if symbolic:
            branch = coef_scale(coef_mul_trig(coef, ch, "s"), w)
        else:
            branch = coef_scale(coef, sval * w)
        if branch:
            out[GP] = coef_add(out.get(GP, {}), branch)
    return out


def _clifford_map(obs: dict, mapping):
    """Apply a Clifford conjugation given as generator images.

    ``mapping`` sends each single-qubit generator (letter, qubit) to a
    (pauli, sign) pair. Composing generator images through :func:`pauli_mul`
    fixes all the signs automatically.
    """
    out = {}
    for P, coef in obs.items():
        x, z = P
        # P = i^{|x&z|} * (prod_q X_q^{x_q}) * (prod_q Z_q^{z_q}). The factors must
        # be mapped in exactly that order: images of different generators need not
        # commute even when the originals do, so interleaving them flips signs.
        acc, ph = (0, 0), 1j ** (_popcount(x & z) % 4)
        for letter, mask in (("X", x), ("Z", z)):
            m = mask
            while m:
                q = (m & -m).bit_length() - 1
                m &= m - 1
                img, s = mapping((letter, q))
                acc, p = pauli_mul(acc, img)
                ph *= p * s
        if abs(ph.imag) > 1e-9:
            raise AssertionError("Clifford image acquired a complex phase")
        out[acc] = coef_add(out.get(acc, {}), coef_scale(coef, ph.real))
    return out


def _cnot_mapping(c, t):
    def m(gen):
        letter, q = gen
        if letter == "X":
            return (single("X", q) if q != c else pauli_mul(single("X", c), single("X", t))[0]), 1.0
        if letter == "Z":
            return (single("Z", q) if q != t else pauli_mul(single("Z", c), single("Z", t))[0]), 1.0
        raise ValueError(gen)
    return m


def _h_mapping(q0):
    def m(gen):
        letter, q = gen
        if q != q0:
            return single(letter, q), 1.0
        return single("Z" if letter == "X" else "X", q), 1.0
    return m


def propagate(spec: EncoderSpec, readout: str, params: dict,
              threshold: float = 1e-10, max_weight: int | None = None,
              max_terms: int | None = None) -> PropagationResult:
    """Propagate ``readout`` backwards through ``spec`` and return the surrogate.

    Parameters
    ----------
    threshold
        Drop Pauli terms whose rigorous coefficient bound falls below this.
    max_weight
        Optionally also drop Pauli strings acting on more than this many qubits.
    max_terms
        Optional hard cap on retained Pauli strings; the smallest are dropped
        first and their bounds accumulate into ``truncation_bound``.
    """
    n = spec.n_qubits
    qs = [int(tok) for tok in readout.replace("Z", " ").split()]
    P0 = (0, 0)
    for q in qs:
        P0, _ = pauli_mul(P0, single("Z", q))
    obs = {P0: {EMPTY: 1.0}}

    res = PropagationResult(readout=readout, n_qubits=n)

    for g in reversed(spec.gates):
        if g[0] == "rot":
            _, axis, q, src = g
            obs = _rot_conjugate(obs, single(axis, q), src, params)
        elif g[0] == "crot":
            # CR_a(θ) = Rot(σ_a,t, θ/2) · Rot(Z_c σ_a,t, −θ/2); the two commute
            _, axis, c, t, src = g
            Ga = single(axis, t)
            Gz, _ = pauli_mul(single("Z", c), Ga)
            obs = _rot_conjugate(obs, Ga, src, params, half=True)
            obs = _neg_angle_conjugate(obs, Gz, src, params)
        elif g[0] == "cnot":
            obs = _clifford_map(obs, _cnot_mapping(g[1], g[2]))
        elif g[0] == "h":
            obs = _clifford_map(obs, _h_mapping(g[1]))
        else:
            raise ValueError(g)

        res.max_pauli_terms = max(res.max_pauli_terms, len(obs))
        obs, dropped, bound = _truncate(obs, threshold, max_weight, max_terms)
        res.dropped += dropped
        res.truncation_bound += bound

    # <0|P|0> is 0 unless P has no X/Y support, and +1 otherwise
    terms: dict = {}
    for (x, z), coef in obs.items():
        if x == 0:
            terms = coef_add(terms, coef)
    res.terms = {m: v for m, v in terms.items() if abs(v) > 1e-14}
    res.n_pauli_terms = len(obs)
    chans = set()
    for m in res.terms:
        for ch, _k, _p in m:
            chans.add(ch)
    res.channels = sorted(chans, key=repr)
    return res


def _neg_angle_conjugate(obs, G, src, params):
    """Conjugate by exp(+i phi G / 4), i.e. a rotation by -phi/2."""
    if _is_data(src):
        neg = _negate_src(_halve_src(src))
        return _rot_conjugate(obs, G, neg, params)
    phi = float(_angle(src, np.zeros((1, 1)), params))
    return _rot_conjugate(obs, G, ("fixed", -phi / 2.0), params)


def _negate_src(src):
    if src[0] == "data":
        return ("data", src[1], -src[2])
    if src[0] == "data_poly":
        return ("data_poly", [(-c, f) for c, f in src[1]])
    raise ValueError(src)


def _truncate(obs, threshold, max_weight, max_terms):
    dropped, bound = 0, 0.0
    keep = {}
    for P, coef in obs.items():
        b = coef_bound(coef)
        if b < threshold or (max_weight is not None and pauli_weight(P) > max_weight):
            dropped += 1
            bound += b
            continue
        keep[P] = coef
    if max_terms is not None and len(keep) > max_terms:
        ranked = sorted(keep.items(), key=lambda kv: coef_bound(kv[1]), reverse=True)
        for P, coef in ranked[max_terms:]:
            keep.pop(P)
            dropped += 1
            bound += coef_bound(coef)
    return keep, dropped, bound


def surrogate(spec: EncoderSpec, params: dict, readouts=None, **kw):
    """Propagate every readout of ``spec``. Returns {readout: PropagationResult}."""
    return {r: propagate(spec, r, params, **kw)
            for r in (readouts if readouts is not None else spec.readouts)}
