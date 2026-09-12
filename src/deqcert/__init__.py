"""Validated classical Fourier surrogates for deployed PQC encoders.

certify() returns a certified surrogate, a surrogate above the requested
tolerance, or an inconclusive result when construction exceeds the budget.
Certification does not assert quantum hardness, family-wide efficiency, or an
absence of other classical simulation methods.
"""

from .certifier import (
    EncoderSpec,
    Report,
    EXAMPLES,
    certify,
    print_report,
    extract_surrogate,
    dla_dimension,
    frequency_axis,
    effective_dimension,
    mean_entanglement,
    readout_support,
    statevector,
    simulate,
)

__all__ = [
    "EncoderSpec",
    "Report",
    "EXAMPLES",
    "certify",
    "print_report",
    "extract_surrogate",
    "dla_dimension",
    "frequency_axis",
    "effective_dimension",
    "mean_entanglement",
    "readout_support",
    "statevector",
    "simulate",
    "from_qiskit",
]

__version__ = "0.2.0"

from .validated import evaluate_surrogate, evaluate_enclosure
__all__ += ["evaluate_surrogate", "evaluate_enclosure"]


def from_qiskit(*args, **kwargs):
    """Build an :class:`EncoderSpec` from a Qiskit ``QuantumCircuit``.

    Requires the optional dependency: ``pip install deqcert[qiskit]``.
    """
    try:
        from .qiskit_import import from_qiskit as _f
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "from_qiskit requires qiskit. Install it with: pip install deqcert[qiskit]"
        ) from exc
    return _f(*args, **kwargs)
