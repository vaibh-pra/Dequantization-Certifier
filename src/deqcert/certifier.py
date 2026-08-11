"""
PQC dequantization certifier with structure-aware exact surrogate extraction.

Given a white-box PQC-encoder spec (and optionally query access to a deployed
instance), report the two simulability axes (dynamical Lie algebra dimension and
encoding-frequency dimension), extract an exact matched-classical surrogate when
one exists, and emit a dequantization verdict.

This file is built up in phases:
  P0  spec + general statevector simulator + example encoders   <-- this commit
  P1  DLA axis (Lie closure)
  P2  encoding-frequency axis + C3
  P3  structure-derived candidate basis + effective dimension
  P4  structure-aware exact surrogate extraction (the core)
  P5  cost / crossover
  P6  verdict + entanglement flag + CLI
  P7  validation suite

Design choices (agreed): hybrid basis (structure-derived candidate + numeric
exact fit + symbolic pruning); exponential-DLA circuits are diagnosed with a
cheap entanglement/depth flag, not solved by a tensor-network backend in v1.
CPU-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# single-qubit matrices
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
PAULI = {
    "X": np.array([[0, 1], [1, 0]], complex),
    "Y": np.array([[0, -1j], [1j, 0]], complex),
    "Z": np.array([[1, 0], [0, -1]], complex),
}
HMAT = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)


def rot(axis: str, theta):
    """R_axis(theta) = exp(-i theta/2 * Pauli_axis). theta may be array (batched)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    z = np.zeros_like(c * 1.0)
    if axis == "X":
        g = np.stack([c + 0j, -1j * s, -1j * s, c + 0j], axis=-1)
    elif axis == "Y":
        g = np.stack([c + 0j, -s + 0j, s + 0j, c + 0j], axis=-1)
    elif axis == "Z":
        g = np.stack([np.exp(-1j * theta / 2), z * 1j, z * 1j, np.exp(1j * theta / 2)],
                     axis=-1)
    else:
        raise ValueError(axis)
    return g.reshape(np.shape(theta) + (2, 2))


# ---------------------------------------------------------------------------
# encoder specification
# ---------------------------------------------------------------------------
# A gate is one of:
#   ("rot", axis, qubit, src)            single-qubit rotation
#   ("crot", axis, control, target, src) controlled single-qubit rotation
#   ("cnot", control, target)            fixed CNOT
# where src (the rotation angle source) is one of:
#   ("data",  feature_index, scale)      angle = scale * x[feature]
#   ("param", name)                      trainable angle (value from a param dict)
#   ("fixed", value)                     fixed angle
@dataclass
class EncoderSpec:
    name: str
    n_qubits: int
    n_features: int
    gates: list
    readouts: list                       # Pauli-Z strings, e.g. "Z0", "Z0Z2"
    note: str = ""

    def param_names(self):
        names = []

        def add(nm):
            if nm not in names:
                names.append(nm)
        for g in self.gates:
            if g[0] in ("rot", "crot"):
                src = g[-1]
                if src[0] == "param":
                    add(src[1])
                elif src[0] == "data" and isinstance(src[2], tuple) and src[2][0] == "param":
                    add(src[2][1])                 # re-encoding scale lambda
        return names

    def scale_params(self):
        sp = set()
        for g in self.gates:
            if g[0] in ("rot", "crot"):
                src = g[-1]
                if src[0] == "data" and isinstance(src[2], tuple):
                    sp.add(src[2][1])
        return sp

    def data_gates(self):
        out = []
        for g in self.gates:
            if g[0] in ("rot", "crot"):
                src = g[-1]
                if src[0] == "data":
                    out.append((g, src))
        return out


# ---------------------------------------------------------------------------
# general batched statevector simulator
# ---------------------------------------------------------------------------
def _apply_1q(psi, g, q, n):
    """g: (2,2) shared or (B,2,2) per-sample, applied to qubit q. psi (B,2^n)."""
    B = psi.shape[0]
    psi = psi.reshape(B, 2 ** q, 2, 2 ** (n - 1 - q))
    if g.ndim == 2:
        psi = np.einsum("xi,baic->baxc", g, psi)
    else:
        psi = np.einsum("bxi,baic->baxc", g, psi)
    return psi.reshape(B, 2 ** n)


def _apply_ctrl_1q(psi, g, c, t, n):
    """Apply (2,2) g to qubit t in the subspace where qubit c == 1."""
    B = psi.shape[0]
    T = psi.reshape((B,) + (2,) * n).copy()
    sl = [slice(None)] * (n + 1)
    sl[c + 1] = 1                                   # control = |1>
    sub = T[tuple(sl)]                              # (B,) + (2,)*(n-1)
    ax = t if t > c else t + 1                      # target axis inside sub
    sub = np.moveaxis(np.tensordot(g, sub, axes=([1], [ax])), 0, ax)
    T[tuple(sl)] = sub
    return T.reshape(B, 2 ** n)


def _cnot_perm(c, t, n):
    idx = np.arange(2 ** n)
    cb, tb = 1 << (n - 1 - c), 1 << (n - 1 - t)
    out = idx.copy()
    m = (idx & cb) > 0
    out[m] = idx[m] ^ tb
    return out


def _angle(src, x, params):
    """Resolve a gate angle. x: (B, n_features); returns scalar or (B,)."""
    kind = src[0]
    if kind == "data":
        _, feat, scale = src
        # scale may be a fixed number or a trainable parameter ("param", name)
        scale_val = params[scale[1]] if isinstance(scale, tuple) else scale
        return scale_val * x[:, feat]
    if kind == "data_poly":
        # angle = sum of coeff * product(x[feat]); e.g. an entangling feature map.
        ang = np.zeros(x.shape[0])
        for coeff, feats in src[1]:
            term = np.full(x.shape[0], float(coeff))
            for f in feats:
                term = term * x[:, f]
            ang = ang + term
        return ang
    if kind == "param":
        return params[src[1]]
    if kind == "fixed":
        return src[1]
    raise ValueError(src)


def _zdiag(pauli_string, n):
    qs = [int(tok) for tok in pauli_string.replace("Z", " ").split()]
    diag = np.ones(2 ** n)
    for q in qs:
        bit = (np.arange(2 ** n) >> (n - 1 - q)) & 1
        diag *= np.where(bit == 0, 1.0, -1.0)
    return diag


def statevector(spec: EncoderSpec, x, params):
    """x: (B, n_features) real; params: {name: value}. Returns psi (B, 2^n)."""
    x = np.atleast_2d(np.asarray(x, float))
    B, n = x.shape[0], spec.n_qubits
    psi = np.zeros((B, 2 ** n), complex)
    psi[:, 0] = 1.0
    for g in spec.gates:
        if g[0] == "rot":
            _, axis, q, src = g
            th = _angle(src, x, params)
            gate = rot(axis, np.full(B, th) if np.ndim(th) == 0 else th)
            psi = _apply_1q(psi, gate, q, n)
        elif g[0] == "crot":
            _, axis, c, t, src = g
            th = _angle(src, x, params)
            if np.ndim(th) == 0:
                psi = _apply_ctrl_1q(psi, rot(axis, th), c, t, n)
            else:                                    # per-sample controlled: loop-free via mask
                gate = rot(axis, th)                 # (B,2,2)
                # apply per-sample by embedding: rare in our specs; do a small loop
                for b in range(B):
                    psi[b:b + 1] = _apply_ctrl_1q(psi[b:b + 1], gate[b], c, t, n)
        elif g[0] == "cnot":
            _, c, t = g
            psi = psi[:, _cnot_perm(c, t, n)]
        elif g[0] == "h":
            psi = _apply_1q(psi, HMAT, g[1], n)
        else:
            raise ValueError(g)
    return psi


def simulate(spec: EncoderSpec, x, params):
    """Returns {readout_string: (B,) expectation values}."""
    psi = statevector(spec, x, params)
    prob = (psi.conj() * psi).real
    return {r: prob @ _zdiag(r, spec.n_qubits) for r in spec.readouts}


# ---------------------------------------------------------------------------
# report container
# ---------------------------------------------------------------------------
@dataclass
class Report:
    name: str
    n_qubits: int = 0
    dla_dim: int | None = None
    dla_class: str = ""          # "polynomial" / "exponential" / "unknown"
    freq_dim: Any = None
    freq_class: str = ""         # "bounded" / "unbounded"
    eff_dim: int | None = None
    surrogate_residual: float | None = None
    basis: list = field(default_factory=list)
    coeffs: Any = None
    cost: dict = field(default_factory=dict)
    verdict: str = ""
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# built-in example encoders
# ---------------------------------------------------------------------------
def hqz_base():
    g = []
    for i in range(4):
        g.append(("rot", "X", i, ("data", i, 1.0)))     # R_X(x_i) encoding
    # pair A: control q1 -> target q0 ; pair B: control q3 -> target q2
    g += [("crot", "Z", 1, 0, ("param", "a10")), ("crot", "X", 1, 0, ("param", "b10")),
          ("crot", "Z", 3, 2, ("param", "a32")), ("crot", "X", 3, 2, ("param", "b32"))]
    return EncoderSpec("hqz_base", 4, 4, g, ["Z0", "Z2"], "base HQZ QCNN encoder")


def hqz_cross():
    s = hqz_base()
    # insert cross gate CRX(gamma): control q2 -> target q0, after pair-A entangler
    gates = s.gates[:6] + [("crot", "X", 2, 0, ("param", "gamma"))] + s.gates[6:]
    return EncoderSpec("hqz_cross", 4, 4, gates, ["Z0", "Z2"],
                       "+CROSS_CRX_q0: cross gate onto the measured qubit")


def reenc_pair(depth=3):
    # single pair (2 qubits), data re-uploaded with trainable lambda_k each layer
    g = [("rot", "X", 0, ("data", 0, 1.0)), ("rot", "X", 1, ("data", 1, 1.0))]
    for k in range(depth):
        if k > 0:
            g += [("rot", "X", 0, ("data", 0, ("param", f"lam{k - 1}"))),
                  ("rot", "X", 1, ("data", 1, ("param", f"lam{k - 1}")))]
        g += [("crot", "Z", 1, 0, ("param", f"a{k}")),
              ("crot", "X", 1, 0, ("param", f"b{k}"))]
    return EncoderSpec("reenc_pair", 2, 2, g, ["Z0"],
                       f"+REENC re-uploading pair, depth {depth}")


def vqc_facedet(layers=3):
    g = [("rot", "X", i, ("data", i, 1.0)) for i in range(8)]
    for L in range(layers):
        for i in range(8):
            g += [("rot", "X", i, ("param", f"x{L}_{i}")),
                  ("rot", "Y", i, ("param", f"y{L}_{i}")),
                  ("rot", "Z", i, ("param", f"z{L}_{i}"))]
        for i in range(8):
            g.append(("cnot", i, (i + 1) % 8))
    return EncoderSpec("vqc_facedet", 8, 8, g, ["Z0", "Z1"],
                       f"8-qubit face-detection VQC, {layers} layers")


def zz_feature_map(n=3, fm_reps=2, var_reps=1):
    """Havlicek et al. (Nature 2019) ZZFeatureMap + RealAmplitudes classifier.
    Uses H gates and the data_poly source for the RZZ(2(pi-x_i)(pi-x_j)) term."""
    pi = math.pi
    g = []
    for _ in range(fm_reps):
        for q in range(n):
            g.append(("h", q))
        for q in range(n):
            g.append(("rot", "Z", q, ("data", q, 2.0)))            # RZ(2 x_q)
        for i in range(n - 1):
            j = i + 1
            theta = ("data_poly", [(2 * pi * pi, []), (-2 * pi, [i]),
                                   (-2 * pi, [j]), (2.0, [i, j])])   # 2(pi-x_i)(pi-x_j)
            g += [("cnot", i, j), ("rot", "Z", j, theta), ("cnot", i, j)]   # RZZ
    k = 0
    for _ in range(var_reps):                                       # RealAmplitudes
        for q in range(n):
            g.append(("rot", "Y", q, ("param", f"w{k}"))); k += 1
        for i in range(n - 1):
            g.append(("cnot", i, i + 1))
    for q in range(n):
        g.append(("rot", "Y", q, ("param", f"w{k}"))); k += 1
    return EncoderSpec("zz_feature_map", n, n, g, [f"Z{q}" for q in range(n)],
                       "Havlicek ZZFeatureMap classifier")


def minimal_qcnn():
    """Minimal 4-qubit QCNN carrying all three QCNN layers:
      - convolution: ONE shared 2-qubit unitary U = CRZ(ac) CRX(bc) on the
        overlapping windows (q1->q0), (q2->q1), (q3->q2) -- shared params make it
        translation invariant;
      - pooling (4->2): controlled rotation CRX(p) with control = pooled qubit,
        target = survivor, on (q1->q0) and (q3->q2). The pooled qubits q1, q3 are
        traced out, which is automatic here since we read out only the survivors
        (deferred-measurement form of measure-and-condition pooling);
      - fully connected: U_f = CRZ(af) CRX(bf) entangling the survivors (q2->q0).
    Data enters once per qubit (single-frequency angle encoding), so no
    re-uploading. Readout on the survivors q0, q2."""
    g = []
    for i in range(4):                                  # encode R_X(x_i)
        g.append(("rot", "X", i, ("data", i, 1.0)))
    for c, t in [(1, 0), (2, 1), (3, 2)]:               # convolution (shared U)
        g.append(("crot", "Z", c, t, ("param", "ac")))
        g.append(("crot", "X", c, t, ("param", "bc")))
    g.append(("crot", "X", 1, 0, ("param", "p")))       # pooling 4->2 (shared p)
    g.append(("crot", "X", 3, 2, ("param", "p")))
    g.append(("crot", "Z", 2, 0, ("param", "af")))      # fully connected (q2->q0)
    g.append(("crot", "X", 2, 0, ("param", "bf")))
    return EncoderSpec("minimal_qcnn", 4, 4, g, ["Z0", "Z2"],
                       note="conv (shared U) + pool (4->2) + FC; single encoding")


def minimal_qcnn_ae():
    """The encoder used in qcnn_autoencoder.py at 4:1 -- same circuit as
    minimal_qcnn but reading <Z0> only (one output per 2x2 patch)."""
    spec = minimal_qcnn()
    spec.name = "minimal_qcnn_ae"
    spec.readouts = ["Z0"]
    spec.note = "qcnn_autoencoder 4:1 encoder (single readout)"
    return spec


EXAMPLES = {"hqz_base": hqz_base, "hqz_cross": hqz_cross,
            "reenc_pair": reenc_pair, "vqc_facedet": vqc_facedet,
            "zz_feature_map": zz_feature_map, "minimal_qcnn": minimal_qcnn,
            "minimal_qcnn_ae": minimal_qcnn_ae}


# ---------------------------------------------------------------------------
# P1: dynamical Lie algebra (full-circuit, all gate generators)
# ---------------------------------------------------------------------------
def _op(pauli_dict, n):
    m = np.array([[1.0 + 0j]])
    for q in range(n):
        m = np.kron(m, PAULI.get(pauli_dict.get(q, "I"), I2) if q in pauli_dict else I2)
    return m


def generators(spec: EncoderSpec):
    """Hermitian generators of every gate (rotations, controlled rotations, CNOTs).
    Controlled-R generator ~ P_t - Z_c P_t; CNOT entangling content ~ Z_c X_t."""
    n = spec.n_qubits
    gens = []
    for g in spec.gates:
        if g[0] == "rot":
            gens.append(_op({g[2]: g[1]}, n))
        elif g[0] == "crot":
            _, axis, c, t, _ = g
            gens.append(_op({t: axis}, n) - _op({c: "Z", t: axis}, n))
        elif g[0] == "cnot":
            _, c, t = g
            gens.append(_op({c: "Z", t: "X"}, n))
        elif g[0] == "h":
            gens.append(_op({g[1]: "X"}, n) + _op({g[1]: "Z"}, n))   # H ~ (X+Z)/sqrt2
    return gens


def dla_dimension(spec: EncoderSpec, cap: int = 64, tol: float = 1e-7):
    """Lie-closure dimension of the full-circuit generators, capped. Returns
    (dim, saturated): saturated=True means it hit the cap (exponential/large)."""
    A = [-1j * g for g in generators(spec)]

    def vec(M):
        v = M.reshape(-1)
        return np.concatenate([v.real, v.imag])

    span_ops, Q = [], []

    def try_add(M):
        v = vec(M)
        w = v.copy()
        for q in Q:
            w -= (q @ v) * q
        nw = np.linalg.norm(w)
        if nw > tol:
            Q.append(w / nw)
            span_ops.append(M)
            return True
        return False

    for a in A:
        try_add(a)
        if len(span_ops) >= cap:
            return cap, True
    changed = True
    while changed:
        changed = False
        for b in list(span_ops):
            for a in A:
                if try_add(a @ b - b @ a):
                    changed = True
                    if len(span_ops) >= cap:
                        return cap, True
    return len(span_ops), False


def _p1_test():
    print("\nP1 full-circuit DLA dimension:")
    expect = {"hqz_base": "so(5)+so(5) = 20", "hqz_cross": "> 20 (pairs coupled)",
              "reenc_pair": "so(5) = 10", "vqc_facedet": "saturates cap -> exponential"}
    for key, make in EXAMPLES.items():
        spec = make()
        d, sat = dla_dimension(spec)
        tag = f">= {d} (saturated, exponential)" if sat else f"{d}"
        print(f"  {key:<12} dim(g) = {tag:<28}  [{expect[key]}]")


# ---------------------------------------------------------------------------
# P2: encoding-frequency axis (the C3 test, structural)
# ---------------------------------------------------------------------------
def frequency_axis(spec: EncoderSpec):
    """How each feature enters the circuit. A feature re-uploaded (>1 time) with a
    trainable scale gives a continuous frequency set -> unbounded (C3 fires)."""
    from collections import defaultdict
    count, param_scale = defaultdict(int), defaultdict(bool)
    for g in spec.gates:
        if g[0] in ("rot", "crot"):
            src = g[-1]
            if src[0] == "data":
                count[src[1]] += 1
                if isinstance(src[2], tuple):
                    param_scale[src[1]] = True
            elif src[0] == "data_poly":
                for _coeff, feats in src[1]:    # fixed coeffs -> bounded
                    for f in feats:
                        count[f] += 1
    unbounded = any(count[f] > 1 and param_scale[f] for f in count)
    desc = {f: ("once" if count[f] == 1
                else f"{count[f]}x" + (" trainable-scale" if param_scale[f] else " fixed-scale"))
            for f in sorted(count)}
    return desc, ("unbounded" if unbounded else "bounded")


# ---------------------------------------------------------------------------
# P3: readout effective dimension (the function-class size the surrogate fits)
# ---------------------------------------------------------------------------
def _sample_params(spec, rng, L):
    sp = spec.scale_params()
    return {nm: (rng.uniform(-L, L) if nm in sp else rng.uniform(0, 2 * np.pi))
            for nm in spec.param_names()}


def effective_dimension(spec: EncoderSpec, P=200, G=200, L=2.0, tol=1e-9, seed=0):
    """SVD rank of the (params x inputs) readout matrix = function-class dimension.
    Capped at min(P, G * n_readouts)."""
    rng = np.random.RandomState(seed)
    Xin = rng.uniform(0, 2 * np.pi, (G, spec.n_features))
    M = np.zeros((P, G * len(spec.readouts)))
    for p in range(P):
        out = simulate(spec, Xin, _sample_params(spec, rng, L))
        M[p] = np.concatenate([out[r] for r in spec.readouts])
    s = np.linalg.svd(M, compute_uv=False)
    rank = int(np.sum(s > tol * s[0]))
    return rank, rank >= min(P, G * len(spec.readouts))


def _p2p3_test():
    print("\nP2 frequency axis + P3 effective dimension:")
    for key, make in EXAMPLES.items():
        spec = make()
        desc, fclass = frequency_axis(spec)
        ed, capped = effective_dimension(spec)
        ed_s = f">= {ed} (capped)" if capped else f"{ed}"
        print(f"  {key:<12} freq={fclass:<10} eff_dim={ed_s:<14} features={desc}")


# ---------------------------------------------------------------------------
# P4: structure-aware exact surrogate extraction
# ---------------------------------------------------------------------------
import itertools


def readout_support(spec, readout, params, eps=1e-4, n_probe=4, seed=0):
    """Features the readout actually depends on, by finite-difference sensitivity."""
    rng = np.random.RandomState(seed)
    dep = set()
    for _ in range(n_probe):
        x = rng.uniform(0, 2 * np.pi, (1, spec.n_features))
        base = simulate(spec, x, params)[readout][0]
        for f in range(spec.n_features):
            xp = x.copy(); xp[0, f] += eps
            if abs(simulate(spec, xp, params)[readout][0] - base) > 1e-7:
                dep.add(f)
    return sorted(dep)


def _src_features(src):
    if src[0] == "data":
        return {src[1]}
    if src[0] == "data_poly":
        return set(f for _c, feats in src[1] for f in feats)
    return set()


def angle_channels(spec, support):
    """The distinct data-encoding ANGLE expressions whose features lie in `support`.
    The surrogate basis is built from cos/sin of these channels (and harmonics), so
    a product feature map (angle = x_i x_j) becomes its own channel. For plain
    single-feature encodings this reduces to the per-feature basis."""
    sup, chans, keys = set(support), [], set()
    for g in spec.gates:
        if g[0] in ("rot", "crot"):
            src = g[-1]
            feats = _src_features(src)
            # skip trainable-scale (re-uploading) channels: that is the unbounded case
            param_scale = src[0] == "data" and isinstance(src[2], tuple)
            if feats and feats <= sup and not param_scale and repr(src) not in keys:
                keys.add(repr(src)); chans.append(src)
    return chans


def _channel_angles(channels, x):
    return [_angle(c, x, {}) for c in channels]      # list of (B,) arrays


def channel_basis(C, K):
    """Products over C channels of {1} U {cos(k.theta), sin(k.theta): k=1..K}."""
    per = [("id", 0)] + [(t, k) for k in range(1, K + 1) for t in ("cos", "sin")]
    return [[(ci, k, t) for ci, (t, k) in enumerate(combo) if t != "id"]
            for combo in itertools.product(per, repeat=C)]


def eval_channel_basis(basis, angles, B):
    Phi = np.ones((B, len(basis)))
    for i, terms in enumerate(basis):
        for ci, k, t in terms:
            a = angles[ci]
            Phi[:, i] *= np.cos(k * a) if t == "cos" else np.sin(k * a)
    return Phi


def extract_surrogate(spec, readout, params, oversample=3, prune=1e-8, Kmax=3,
                      max_basis=8000, seed=0):
    """Fit the deployed readout on the channel basis, growing the harmonic order K
    until the held-out residual vanishes. Returns the explicit (pruned) surrogate."""
    rng = np.random.RandomState(seed)
    S = readout_support(spec, readout, params, seed=seed)
    channels = angle_channels(spec, S)
    Xt = rng.uniform(0, 2 * np.pi, (300, spec.n_features))
    ft = simulate(spec, Xt, params)[readout]
    angt = _channel_angles(channels, Xt)
    result = None
    for K in range(1, Kmax + 1):
        basis = channel_basis(len(channels), K)
        Bn = len(basis)
        if Bn > max_basis:
            break
        Xq = rng.uniform(0, 2 * np.pi, (oversample * Bn + 5, spec.n_features))
        Phi = eval_channel_basis(basis, _channel_angles(channels, Xq), Xq.shape[0])
        c, *_ = np.linalg.lstsq(Phi, simulate(spec, Xq, params)[readout], rcond=None)
        residual = float(np.max(np.abs(eval_channel_basis(basis, angt, 300) @ c - ft)))
        terms = [(basis[i], float(c[i])) for i in range(Bn) if abs(c[i]) > prune]
        result = dict(readout=readout, support=S, channels=channels, harmonic_K=K,
                      n_channels=len(channels), n_candidates=Bn, queries=len(Xq),
                      n_terms=len(terms), residual=residual, terms=terms)
        if residual < 1e-8:
            break
    return result


def _fmt_term(terms_coeff):
    terms, c = terms_coeff
    label = "1" if not terms else "*".join(f"{t}({k}c{ci})" for ci, k, t in terms)
    return f"{c:+.4f} {label}"


def _p4_test():
    print("\nP4 structure-aware exact surrogate extraction (deployed instances):")
    rng = np.random.RandomState(1)
    for key in ("hqz_base", "hqz_cross"):
        spec = EXAMPLES[key]()
        params = {nm: rng.uniform(0, 2 * np.pi) for nm in spec.param_names()}
        r = extract_surrogate(spec, "Z0", params)
        print(f"  {key:<10} Z0: support={r['support']} candidates={r['n_candidates']} "
              f"queries={r['queries']} terms={r['n_terms']} residual={r['residual']:.2e}")
        if key == "hqz_base":
            print("           surrogate:  z0 = "
                  + "  ".join(_fmt_term(t) for t in r["terms"]))


# ---------------------------------------------------------------------------
# P5: entanglement heuristic (tractability flag for the algebra-escape case)
# ---------------------------------------------------------------------------
def mean_entanglement(spec, cut=None, n_samples=8, L=2.0, seed=0):
    """Mean bipartite entanglement entropy (bits) across a cut, over random
    params/inputs. Low vs the max signals tensor-network simulability."""
    rng = np.random.RandomState(seed)
    n = spec.n_qubits
    cut = cut if cut is not None else n // 2
    ents = []
    for _ in range(n_samples):
        x = rng.uniform(0, 2 * np.pi, (1, spec.n_features))
        psi = statevector(spec, x, _sample_params(spec, rng, L))[0]
        s = np.linalg.svd(psi.reshape(2 ** cut, 2 ** (n - cut)), compute_uv=False)
        p = (s ** 2)
        p = p[p > 1e-12]
        ents.append(float(-np.sum(p * np.log2(p))))
    return np.mean(ents), float(min(cut, n - cut))


# ---------------------------------------------------------------------------
# P6: the certifier (combine axes -> verdict) + report
# ---------------------------------------------------------------------------
def certify(spec: EncoderSpec, deployed_params=None, cap=64, P=200, G=200, seed=0):
    rep = Report(name=spec.name, n_qubits=spec.n_qubits)
    rep.dla_dim, dla_sat = dla_dimension(spec, cap=cap)
    rep.dla_class = "exponential (>= cap)" if dla_sat else "small/polynomial"
    fdesc, rep.freq_class = frequency_axis(spec)
    rep.eff_dim, eff_capped = effective_dimension(spec, P=P, G=G, seed=seed)
    rep.notes.append(f"features: {fdesc}")
    rep.notes.append(f"full-circuit DLA dim = {rep.dla_dim} ({rep.dla_class}); "
                     "note: a large full DLA does not by itself prevent dequantization, "
                     "the readout function class governs that.")

    if rep.freq_class == "unbounded":
        rep.verdict = "ESCAPED via the encoding-frequency axis (data re-uploading)"
        rep.notes.append("Function class leaves any finite basis as the trainable scales "
                         "range; the matched-classical basis grows with the re-encoding "
                         "range (located crossover d ~ 5-7).")
        rep.notes.append("A DEPLOYED instance at fixed scales is still exactly surrogatable "
                         "on its realized finite frequency set.")
    elif eff_capped:
        rep.verdict = "ESCAPED via the dynamical Lie algebra (expressive ansatz)"
        ent, entmax = mean_entanglement(spec, seed=seed)
        rep.notes.append(f"Readout function class is large (eff dim >= {rep.eff_dim}); "
                         "no small matched-classical basis.")
        if ent < 0.5 * entmax:
            rep.notes.append(f"BUT low entanglement ({ent:.2f}/{entmax:.0f} bits across the "
                             "cut) => likely tensor-network simulable; not a quantum win.")
        else:
            rep.notes.append(f"High entanglement ({ent:.2f}/{entmax:.0f} bits) => genuinely "
                             "expensive, and barren-plateau-prone (untrainable at scale).")
    else:
        if deployed_params is None:
            rngp = np.random.RandomState(seed)
            deployed_params = _sample_params(spec, rngp, L=1.0)
        Bsum, maxres = 0, 0.0
        for r in spec.readouts:
            sr = extract_surrogate(spec, r, deployed_params, seed=seed)
            Bsum += sr["n_terms"]
            maxres = max(maxres, sr["residual"])
            rep.basis.append(sr)
            rep.notes.append(f"surrogate[{r}]: {sr['n_terms']} terms, "
                             f"{sr['queries']} queries, residual {sr['residual']:.2e}")
        rep.eff_dim = Bsum
        rep.surrogate_residual = maxres
        if maxres > 1e-6:                       # residual-gated verdict (honest)
            rep.verdict = ("NOT dequantizable in the per-feature trigonometric basis "
                           f"(surrogate residual {maxres:.1e})")
            rep.notes.append("The encoding produces frequencies outside the "
                             "{1, cos x_i, sin x_i} per-feature basis (e.g. a product / "
                             "data-entangling feature map). A cross-frequency basis "
                             "extension is needed to certify or refute dequantization.")
        else:
            rep.verdict = "DEQUANTIZABLE (bounded function class)"
            rep.cost = {"classical_terms_B": Bsum, "statevector_dim_2^n": 2 ** spec.n_qubits,
                        "verdict": "classical O(B) eval, no shots; quantum pays O(2^n) "
                                   "(sim) or O(shots) (QPU)"}
    return rep


def print_report(rep: Report):
    line = "=" * 72
    print(f"\n{line}\n  DEQUANTIZATION CERTIFICATE: {rep.name}  (n = {rep.n_qubits} qubits)\n{line}")
    print(f"  VERDICT:  {rep.verdict}")
    if rep.surrogate_residual is not None:
        print(f"  surrogate basis size B = {rep.eff_dim}   certified residual = "
              f"{rep.surrogate_residual:.2e}")
        if rep.cost:
            print(f"  cost: classical O({rep.cost['classical_terms_B']}) vs "
                  f"statevector O({rep.cost['statevector_dim_2^n']})")
    else:
        print(f"  effective dimension ~ {rep.eff_dim}    freq axis: {rep.freq_class}")
    for nt in rep.notes:
        print(f"    - {nt}")
    if rep.basis and rep.basis[0]["n_terms"] <= 12:
        b0 = rep.basis[0]
        print(f"  explicit surrogate [{b0['readout']}]:  "
              + "  ".join(_fmt_term(t) for t in b0["terms"]))


# ---------------------------------------------------------------------------
# P0 smoke test: simulator vs the HQZ closed form
# ---------------------------------------------------------------------------
def _closed_form_z0(x0, x1, a, b):
    return (np.cos(b / 2) ** 2 * np.cos(x0)
            + np.sin(b / 2) ** 2 * np.cos(x0) * np.cos(x1)
            - 0.5 * np.sin(b) * np.cos(a) * np.sin(x0)
            + 0.5 * np.sin(b) * np.cos(a) * np.sin(x0) * np.cos(x1))


def _p0_smoke():
    rng = np.random.RandomState(0)
    spec = hqz_base()
    maxerr = 0.0
    for _ in range(200):
        x = rng.uniform(0, 2 * np.pi, (1, 4))
        a10, b10 = rng.uniform(0, 2 * np.pi, 2)
        params = {"a10": a10, "b10": b10, "a32": rng.uniform(0, 2 * np.pi),
                  "b32": rng.uniform(0, 2 * np.pi)}
        z0 = simulate(spec, x, params)["Z0"][0]
        ref = _closed_form_z0(x[0, 0], x[0, 1], a10, b10)
        maxerr = max(maxerr, abs(z0 - ref))
    print(f"P0 simulator vs HQZ closed form: max |z0_sim - z0_closed| = {maxerr:.2e}")
    # cross gate sanity: z0 should now depend on x2
    sc = hqz_cross()
    base_x = np.array([[0.5, 2.0, 0.0, 1.4]])
    pr = {"a10": 0.7, "b10": 1.1, "a32": -0.4, "b32": 0.9, "gamma": 1.3}
    z0a = simulate(sc, base_x, pr)["Z0"][0]
    base_x2 = base_x.copy(); base_x2[0, 2] = 3.0
    z0b = simulate(sc, base_x2, pr)["Z0"][0]
    print(f"P0 cross gate sees x2: |dz0| = {abs(z0a - z0b):.4f} (nonzero expected)")
    print("P0 OK" if maxerr < 1e-12 else "P0 CHECK")
