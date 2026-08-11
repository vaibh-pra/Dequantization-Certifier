"""
Qiskit importer for the dequantization certifier.

Walks a qiskit QuantumCircuit and emits an EncoderSpec. You must tell it which
Parameters are the DATA features (everything else is treated as a trainable
angle). Supported gates: rx/ry/rz, crx/cry/crz, cx (CNOT); barriers/measures are
skipped. Angle forms recognised per gate:
    x_i                 -> ("data", i, 1.0)
    c * x_i             -> ("data", i, c)
    lambda * x_i        -> ("data", i, ("param","lambda"))   # data re-uploading
    theta               -> ("param", "theta")
    <number>            -> ("fixed", value)

Run the self-test:  python qiskit_import.py
"""
from .certifier import EncoderSpec, certify, print_report

_ROT = {"rx": ("rot", "X"), "ry": ("rot", "Y"), "rz": ("rot", "Z"),
        "crx": ("crot", "X"), "cry": ("crot", "Y"), "crz": ("crot", "Z")}
_SKIP = {"barrier", "measure", "id", "delay", "snapshot"}


def from_qiskit(qc, data_params, readouts, name="imported"):
    """qc: QuantumCircuit; data_params: ordered list of qiskit Parameters that are
    the input features x_0..x_{d-1}; readouts: Pauli-Z strings e.g. ["Z0","Z2"]."""
    data_index = {p: i for i, p in enumerate(data_params)}

    def qidx(q):
        return qc.find_bit(q).index

    def parse(expr):
        try:                                   # plain number?
            return ("fixed", float(expr))
        except (TypeError, ValueError):
            pass
        params = list(getattr(expr, "parameters", []))
        data_ps = [p for p in params if p in data_index]
        train_ps = [p for p in params if p not in data_index]
        if not data_ps:
            if len(train_ps) == 1:
                return ("param", train_ps[0].name)
            raise NotImplementedError(f"unsupported trainable angle: {expr}")
        if len(data_ps) == 1:
            pd, feat = data_ps[0], data_index[data_ps[0]]
            if not train_ps:                   # scale * x : recover the scale
                hi = float(expr.bind({pd: 1.0}))
                lo = float(expr.bind({pd: 0.0}))
                return ("data", feat, hi - lo)
            if len(train_ps) == 1:             # lambda * x : re-uploading
                return ("data", feat, ("param", train_ps[0].name))
        raise NotImplementedError(f"unsupported angle expression: {expr}")

    gates = []
    for ci in qc.data:
        op = ci.operation
        nm = op.name
        qs = [qidx(q) for q in ci.qubits]
        if nm in _SKIP:
            continue
        if nm == "h":
            gates.append(("h", qs[0]))
        elif nm == "cx":
            gates.append(("cnot", qs[0], qs[1]))
        elif nm in _ROT:
            kind, axis = _ROT[nm]
            src = parse(op.params[0])
            gates.append((kind, axis, qs[0], src) if kind == "rot"
                         else (kind, axis, qs[0], qs[1], src))
        else:
            raise NotImplementedError(
                f"gate '{nm}' not supported; decompose to rx/ry/rz/crx/cry/crz/cx")
    return EncoderSpec(name, qc.num_qubits, len(data_params), gates,
                       list(readouts), note="imported from qiskit")


# ---------------------------------------------------------------------------
def _selftest():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector

    # (1) HQZ-equivalent circuit -> should certify DEQUANTIZABLE
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

    # (2) re-uploading circuit with trainable scale -> should ESCAPE (frequency)
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
