"""Conservative symbolic Qiskit importer.

Accepted data expressions are polynomials, and mixed expressions are exactly
lambda*x. Trainable-only angles must be bare parameters. Unsupported forms,
measurements, resets and classical control are rejected rather than guessed.
Constants are normalized to binary64; certification refers to that normalized
EncoderSpec, not an unrounded symbolic source expression or noisy hardware.
"""
from .certifier import EncoderSpec, certify, print_report

_ROT = {"rx": ("rot", "X"), "ry": ("rot", "Y"), "rz": ("rot", "Z"),
        "p": ("rot", "Z"),                       # phase gate: rz up to global phase
        "crx": ("crot", "X"), "cry": ("crot", "Y"), "crz": ("crot", "Z")}
_FIXED = {"s": ("Z", 0.5), "sdg": ("Z", -0.5), "sx": ("X", 0.5), "sxdg": ("X", -0.5),
          "t": ("Z", 0.25), "tdg": ("Z", -0.25)}   # axis, angle in units of pi
_SKIP = {"barrier", "id"}


def from_qiskit(qc, data_params, readouts, name="imported"):
    """qc: QuantumCircuit; data_params: ordered list of qiskit Parameters that are
    the input features x_0..x_{d-1}; readouts: Pauli-Z strings e.g. ["Z0","Z2"]."""
    data_index = {p: i for i, p in enumerate(data_params)}

    def qidx(q):
        return qc.find_bit(q).index

    def parse(expr):
        import math
        import sympy as sp
        try:
            value = float(expr)
            if not math.isfinite(value):
                raise ValueError("non-finite angle")
            return ("fixed", value)
        except TypeError:
            pass
        params = list(getattr(expr, "parameters", []))
        if not params:
            raise NotImplementedError(f"unsupported angle: {expr}")
        symbolic = sp.sympify(expr.sympify())
        data_ps = [p for p in params if p in data_index]
        train_ps = [p for p in params if p not in data_index]
        symbols = {p: sp.Symbol(p.name) for p in params}
        if not data_ps:
            if len(train_ps) == 1 and sp.expand(symbolic - symbols[train_ps[0]]) == 0:
                return ("param", train_ps[0].name)
            raise NotImplementedError(f"trainable angle must be a bare parameter; bind or decompose: {expr}")
        if train_ps:
            if len(data_ps) == len(train_ps) == 1 and sp.expand(
                    symbolic - symbols[data_ps[0]]*symbols[train_ps[0]]) == 0:
                return ("data", data_index[data_ps[0]], ("param", train_ps[0].name))
            raise NotImplementedError(f"unsupported mixed data/parameter angle: {expr}")
        ordered = sorted(data_ps, key=lambda p: data_index[p])
        try:
            poly = sp.Poly(symbolic, *(symbols[p] for p in ordered))
        except sp.PolynomialError as exc:
            raise NotImplementedError(f"data angle must be a polynomial: {expr}") from exc
        terms = []
        for powers, coeff in poly.terms():
            value = float(coeff)
            if not math.isfinite(value):
                raise ValueError("non-finite polynomial coefficient")
            if value == 0 and coeff != 0:
                raise ValueError("polynomial coefficient underflows binary64")
            if coeff != 0:
                features = [data_index[p] for p, degree in zip(ordered, powers) for _ in range(degree)]
                terms.append((value, features))
        if len(terms) == 1 and len(terms[0][1]) == 1:
            return ("data", terms[0][1][0], terms[0][0])
        return ("data_poly", terms)

    gates = []
    for ci in qc.data:
        op = ci.operation
        if getattr(op, "condition", None) is not None:
            raise NotImplementedError("classically conditioned operations are unsupported")
        nm = op.name
        qs = [qidx(q) for q in ci.qubits]
        if nm in _SKIP:
            continue
        if nm == "h":
            gates.append(("h", qs[0]))
        elif nm == "cx":
            gates.append(("cnot", qs[0], qs[1]))
        elif nm in _FIXED:
            axis, frac = _FIXED[nm]
            import math
            gates.append(("rot", axis, qs[0], ("fixed", frac * math.pi)))
        elif nm in _ROT:
            kind, axis = _ROT[nm]
            src = parse(op.params[0])
            gates.append((kind, axis, qs[0], src) if kind == "rot"
                         else (kind, axis, qs[0], qs[1], src))
        else:
            raise NotImplementedError(
                f"gate '{nm}' not supported; decompose to rx/ry/rz/crx/cry/crz/cx")
    return EncoderSpec(name, qc.num_qubits, len(data_params), gates,
                       list(readouts), note="symbolically parsed from Qiskit; certificate applies to normalized binary64 gate specification")


# ---------------------------------------------------------------------------
def _selftest():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector

    # (1) HQZ-equivalent circuit: certify the recorded deployed instance.
    x = ParameterVector("x", 4)
    a10, b10, a32, b32 = (Parameter(n) for n in ("a10", "b10", "a32", "b32"))
    qc = QuantumCircuit(4)
    for i in range(4):
        qc.rx(x[i], i)
    qc.crz(a10, 1, 0); qc.crx(b10, 1, 0)
    qc.crz(a32, 3, 2); qc.crx(b32, 3, 2)
    spec = from_qiskit(qc, list(x), ["Z0", "Z2"], "hqz_from_qiskit")
    print(f"imported HQZ: {len(spec.gates)} gates, params={spec.param_names()}")
    print_report(certify(spec))

    # (2) Freeze the trainable scale and certify that deployed instance.
    x2 = ParameterVector("x", 2)
    lam = Parameter("lam")
    a, b = ParameterVector("a", 2), ParameterVector("b", 2)
    qc2 = QuantumCircuit(2)
    qc2.rx(x2[0], 0); qc2.rx(x2[1], 1)
    qc2.crz(a[0], 1, 0); qc2.crx(b[0], 1, 0)
    qc2.rx(lam * x2[0], 0); qc2.rx(lam * x2[1], 1)     # re-upload, trainable scale
    qc2.crz(a[1], 1, 0); qc2.crx(b[1], 1, 0)
    spec2 = from_qiskit(qc2, list(x2), ["Z0"], "reupload_from_qiskit")
    print(f"\nimported re-uploading: scale_params={spec2.scale_params()}")
    print_report(certify(spec2))


if __name__ == "__main__":
    _selftest()
