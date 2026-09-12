"""Validated Fourier reconstruction on an independent encoding-channel torus.

Only elementary mpmath interval operations are used. The certificate bounds the
mathematical surrogate (with the returned binary64 coefficients interpreted
exactly), not arbitrary floating-point evaluation of its phases at user inputs.
"""
from __future__ import annotations

import itertools
import math
import re
import time
from dataclasses import replace

import numpy as np
from mpmath.ctx_iv import MPIntervalContext


def validate_spec(spec, params):
    if not isinstance(spec.n_qubits, int) or spec.n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")
    if not isinstance(spec.n_features, int) or spec.n_features < 0:
        raise ValueError("n_features must be a nonnegative integer")
    if not spec.readouts:
        raise ValueError("at least one readout is required")

    def finite(v):
        if not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(v):
            raise ValueError("angle constants and deployed parameters must be finite real numbers")
        if float(v) != v:
            raise ValueError("normalize angle constants and deployed parameters to binary64 explicitly")

    def feature(f):
        if not isinstance(f, int) or not 0 <= f < spec.n_features:
            raise ValueError("invalid feature index")

    for r in spec.readouts:
        if not isinstance(r, str) or not re.fullmatch(r"(?:Z\d+)+", r):
            raise ValueError("readouts must be Pauli-Z products, such as Z0Z2")
        qs = [int(q) for q in re.findall(r"Z(\d+)", r)]
        if len(set(qs)) != len(qs) or any(q >= spec.n_qubits for q in qs):
            raise ValueError("readout qubits must be distinct and in range")
    for g in spec.gates:
        if not g or g[0] not in ("rot", "crot", "cnot", "h"):
            raise ValueError(f"unsupported gate: {g}")
        kind = g[0]
        length = {"rot": 4, "crot": 5, "cnot": 3, "h": 2}[kind]
        if len(g) != length:
            raise ValueError("malformed gate")
        qs = g[2:-1] if kind in ("rot", "crot") else g[1:]
        if len(set(qs)) != len(qs) or any(not isinstance(q, int) or not 0 <= q < spec.n_qubits for q in qs):
            raise ValueError("invalid gate qubits")
        if kind not in ("rot", "crot"):
            continue
        if g[1] not in ("X", "Y", "Z"):
            raise ValueError("unsupported rotation axis")
        src = g[-1]
        if src[0] == "data" and len(src) == 3:
            feature(src[1])
            scale = src[2]
            if isinstance(scale, tuple):
                if len(scale) != 2 or scale[0] != "param" or scale[1] not in params:
                    raise ValueError("missing deployed scale parameter")
                finite(params[scale[1]])
            else:
                finite(scale)
        elif src[0] == "data_poly" and len(src) == 2:
            for coeff, feats in src[1]:
                finite(coeff)
                for f in feats:
                    feature(f)
        elif src[0] in ("param", "fixed") and len(src) == 2:
            if src[0] == "param" and src[1] not in params:
                raise ValueError("missing deployed angle parameter")
            finite(params[src[1]] if src[0] == "param" else src[1])
        else:
            raise ValueError(f"unsupported angle source: {src}")


def lightcone(spec, readout):
    """Conservative reverse causal cone: no numerical sensitivity pruning."""
    active = {int(q) for q in re.findall(r"Z(\d+)", readout)}
    kept = []
    for gate in reversed(spec.gates):
        qs = set(gate[2:-1] if gate[0] in ("rot", "crot") else gate[1:])
        if active & qs:
            kept.append(gate)
            active |= qs
    mapping = {q: i for i, q in enumerate(sorted(active))}
    gates = []
    for gate in reversed(kept):
        if gate[0] in ("rot", "crot"):
            gates.append((gate[0], gate[1], *(mapping[q] for q in gate[2:-1]), gate[-1]))
        else:
            gates.append((gate[0], *(mapping[q] for q in gate[1:])))
    r = "".join(f"Z{mapping[int(q)]}" for q in re.findall(r"Z(\d+)", readout))
    return replace(spec, n_qubits=len(active), gates=gates, readouts=[r])


def frozen_channels(spec, params):
    channels, keys, counts, half = [], [], [], []
    gate_channel = {}
    for i, gate in enumerate(spec.gates):
        if gate[0] not in ("rot", "crot"):
            continue
        src = gate[-1]
        if src[0] not in ("data", "data_poly"):
            continue
        if src[0] == "data" and isinstance(src[2], tuple):
            src = ("data", src[1], float(params[src[2][1]]))
        # Keeping algebraically identical expressions as separate channels is
        # conservative: the lifted torus contains the physical channel image.
        key = repr(src)
        if key not in keys:
            keys.append(key)
            channels.append(src)
            counts.append(0)
            half.append(False)
        c = keys.index(key)
        counts[c] += 1
        half[c] |= gate[0] == "crot"
        gate_channel[i] = c
    return channels, counts, half, gate_channel


def plan_readout(spec, readout, params, max_basis=8000, max_active_qubits=6,
                 max_work=20_000_000, max_harmonic=32):
    cone = lightcone(spec, readout)
    channels, counts, half, gate_channel = frozen_channels(cone, params)
    orders = [r * (2 if h else 1) for r, h in zip(counts, half)]
    sizes = [2*k + 1 for k in orders]
    points = math.prod(sizes)
    work = points * (2**cone.n_qubits * max(1, len(cone.gates)) + sum(sizes))
    reasons = []
    if points > max_basis:
        reasons.append(f"grid {points} > {max_basis}")
    if cone.n_qubits > max_active_qubits:
        reasons.append(f"active qubits {cone.n_qubits} > {max_active_qubits}")
    if work > max_work:
        reasons.append(f"work estimate {work} > {max_work}")
    if max(orders, default=0) > max_harmonic:
        reasons.append(f"harmonic order exceeds {max_harmonic}")
    return dict(cone=cone, channels=channels, counts=counts, half=half,
                gate_channel=gate_channel, orders=orders, sizes=sizes,
                points=points, work=work, reasons=reasons)


def _matrix(iv, axis, angle):
    c, s = iv.cos(angle/2), iv.sin(angle/2)
    if axis == "X":
        return (c, -iv.j*s, -iv.j*s, c)
    if axis == "Y":
        return (c, -s, s, c)
    return (iv.exp(-iv.j*angle/2), iv.mpc(0), iv.mpc(0), iv.exp(iv.j*angle/2))


def _gate_cache(iv, plan, params):
    cache = []
    for gi, gate in enumerate(plan["cone"].gates):
        if gate[0] in ("rot", "crot"):
            if gi in plan["gate_channel"]:
                c = plan["gate_channel"][gi]
                period = 4 if plan["half"][c] else 2
                mats = [_matrix(iv, gate[1], period*iv.pi*j/plan["sizes"][c])
                        for j in range(plan["sizes"][c])]
                cache.append((gate, c, mats))
            else:
                src = gate[-1]
                angle = params[src[1]] if src[0] == "param" else src[1]
                cache.append((gate, None, [_matrix(iv, gate[1], iv.mpf(float(angle)))]))
        elif gate[0] == "h":
            h = 1 / iv.sqrt(iv.mpf(2))
            cache.append((gate, None, [(h, h, h, -h)]))
        else:
            cache.append((gate, None, []))
    return cache


def _sample(iv, plan, cache, index):
    n = plan["cone"].n_qubits
    psi = [iv.mpc(0) for _ in range(2**n)]
    psi[0] = iv.mpc(1)
    for gate, channel, mats in cache:
        kind = gate[0]
        if kind == "cnot":
            cm, tm = 1 << (n-1-gate[1]), 1 << (n-1-gate[2])
            for i in range(2**n):
                if i & cm and not i & tm:
                    psi[i], psi[i | tm] = psi[i | tm], psi[i]
            continue
        mat = mats[index[channel] if channel is not None else 0]
        target = gate[1] if kind == "h" else gate[-2]
        tm = 1 << (n-1-target)
        cm = 1 << (n-1-gate[2]) if kind == "crot" else 0
        for i in range(2**n):
            if i & tm or (cm and not i & cm):
                continue
            a, b = psi[i], psi[i | tm]
            psi[i] = mat[0]*a + mat[1]*b
            psi[i | tm] = mat[2]*a + mat[3]*b
    qs = [int(q) for q in re.findall(r"Z(\d+)", plan["cone"].readouts[0])]
    total = iv.mpf(0)
    for i, z in enumerate(psi):
        sign = -1 if sum((i >> (n-1-q)) & 1 for q in qs) % 2 else 1
        total += sign * (z.real*z.real + z.imag*z.imag)
    return iv.mpc(total)


def _tensor_dft(iv, values, sizes):
    """Separable direct DFT, with interval roots of unity and normalisation."""
    a = np.array(values, dtype=object).reshape(tuple(sizes) or ())
    for axis, size in enumerate(sizes):
        v = np.moveaxis(a, axis, -1).copy()
        roots = [[iv.exp(-2*iv.pi*iv.j*k*j/size)/size for j in range(size)]
                 for k in range(size)]
        for ix in np.ndindex(v.shape[:-1]):
            row = list(v[ix])
            v[ix] = [sum((roots[k][j]*row[j] for j in range(size)), iv.mpc(0))
                     for k in range(size)]
        a = np.moveaxis(v, -1, axis)
    return a


def _upper_float(interval):
    # Float conversion rounds to nearest; stepping toward +inf preserves an
    # upper enclosure, including when the exact endpoint is between floats.
    return math.nextafter(float(interval.b), math.inf)


def reconstruct(plan, params, readout, tolerance=1e-10, prune=1e-14, dps=40):
    if plan["reasons"]:
        raise ValueError("cannot reconstruct an over-budget plan")
    if not math.isfinite(tolerance) or tolerance <= 0 or not math.isfinite(prune) or prune < 0:
        raise ValueError("tolerance must be positive and prune nonnegative")
    if not isinstance(dps, int) or dps < 20:
        raise ValueError("interval precision must be at least 20 decimal digits")
    start = time.perf_counter()
    iv = MPIntervalContext()
    iv.dps = dps
    cache = _gate_cache(iv, plan, params)
    indices = list(itertools.product(*(range(s) for s in plan["sizes"])))
    values = [_sample(iv, plan, cache, ix) for ix in indices]
    coefficients = _tensor_dft(iv, values, plan["sizes"])
    terms, bound, unpruned_bound, pruned_bound = [], iv.mpf(0), iv.mpf(0), iv.mpf(0)
    for ix in indices:
        box = coefficients[ix] if ix else coefficients.item()
        center = complex(float(box.real.mid), float(box.imag.mid))
        radius = abs(box.real-iv.mpf(center.real)) + abs(box.imag-iv.mpf(center.imag))
        unpruned_bound += radius
        if abs(center) <= prune:
            error = abs(box.real) + abs(box.imag)
            pruned_bound += error
        else:
            error = radius
            freq = [(k if k <= order else k-size) / (2 if half else 1)
                    for k, order, size, half in zip(ix, plan["orders"], plan["sizes"], plan["half"])]
            terms.append((freq, center.real, center.imag))
        bound += error
    eps = _upper_float(bound)
    if not math.isfinite(eps):
        eps = math.inf
    return dict(readout=readout, channels=plan["channels"], counts=plan["counts"],
                half_angle=plan["half"], grid_sizes=plan["sizes"],
                n_candidates=plan["points"], queries=plan["points"], n_terms=len(terms),
                terms=terms, coefficient_error_bound=eps,
                unpruned_error_bound=_upper_float(unpruned_bound),
                discarded_error_bound=_upper_float(pruned_bound),
                tolerance=tolerance, certified=eps <= tolerance, interval_dps=dps,
                representation="complex_fourier", active_qubits=plan["cone"].n_qubits,
                construction_time_s=time.perf_counter()-start,
                guarantee="global mathematical error for normalized spec at fixed deployed parameters")


def evaluate_surrogate(surrogate, x):
    """Fast approximate evaluation; its own rounding error is not certified."""
    from .certifier import _channel_angles
    x = np.atleast_2d(np.asarray(x, float))
    angles = _channel_angles(surrogate["channels"], x)
    result = np.zeros(len(x))
    for freq, re_, im_ in surrogate["terms"]:
        phase = sum((k*a for k, a in zip(freq, angles)), np.zeros(len(x)))
        result += re_*np.cos(phase) - im_*np.sin(phase)
    return result


def evaluate_enclosure(surrogate, x, dps=40):
    """Return per-input interval enclosures including construction AND evaluation.

    Input floats represent exact binary numbers. Endpoints are exported outward
    to binary64; polynomial phase evaluation and trig use interval arithmetic.
    """
    iv = MPIntervalContext()
    iv.dps = dps
    out = []
    for row in np.atleast_2d(np.asarray(x, float)):
        angles = []
        for src in surrogate["channels"]:
            if src[0] == "data":
                angles.append(iv.mpf(float(src[2]))*iv.mpf(float(row[src[1]])))
            else:
                a = iv.mpf(0)
                for coeff, feats in src[1]:
                    term = iv.mpf(float(coeff))
                    for f in feats:
                        term *= iv.mpf(float(row[f]))
                    a += term
                angles.append(a)
        val = iv.mpf(0)
        for freq, re_, im_ in surrogate["terms"]:
            phase = sum((iv.mpf(k)*a for k, a in zip(freq, angles)), iv.mpf(0))
            val += iv.mpf(re_)*iv.cos(phase) - iv.mpf(im_)*iv.sin(phase)
        eps = surrogate["coefficient_error_bound"]
        val += iv.mpf([-eps, eps])
        out.append((math.nextafter(float(val.a), -math.inf), _upper_float(val)))
    return np.array(out)
