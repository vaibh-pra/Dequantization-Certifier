# Changelog

## 0.2.0 — 12 September 2026

Certification now means a uniform absolute error bound for a supported normalized circuit at fixed deployed parameters. Previous fit-residual-based verdicts must not be interpreted as numerical certificates under this contract.

- Replace residual-based acceptance with complete lifted-channel Fourier reconstruction and interval coefficient bounds.
- Preserve weak input dependencies through structural causal cones and handle dependent angle expressions without a physical-input rank assumption.
- Include controlled-rotation half frequencies, coefficient rounding and pruning in the guarantee.
- Freeze trainable scales at their recorded deployed values.
- Return an inconclusive outcome before evaluation when a readout exceeds the resource policy.
- Parse supported Qiskit polynomial expressions symbolically and reject unsupported expressions or measurements.
- Add fast surrogate evaluation and validated pointwise enclosures.
- Make algebra, rank and entanglement diagnostics optional and independent of certification.
- Add analytical and independent-Qiskit regression checks, a standalone benchmark driver and recorded circuit results.

### Migration

Pass `deployed_params` to certify a chosen deployment. Omitting it selects and records a seeded demonstration instance.

`report.basis` now contains complex Fourier representations. Use `evaluate_surrogate` for fast numerical evaluation or `evaluate_enclosure` for pointwise enclosures. Fast evaluation has additional rounding error beyond the mathematical construction bound.

`report.certificate_error` is the largest readout bound. The legacy `eff_dim` field now counts retained complex Fourier modes; prefer `report.cost['retained_fourier_modes']`. It is not a function-class dimension.

The outcomes are `CERTIFIED epsilon surrogate`, `SURROGATE FOUND (requested error tolerance not met)`, and `INCONCLUSIVE (resource budget)`. Invalid inputs raise errors. Manual `K` overrides cannot produce certificates, and `extract_surrogate` is exploratory with `certified=False`.

The numerical guarantee depends on the correctness of the elementary interval operations. It does not establish family-wide efficiency, optimal sparsity, quantum hardness, hardware accuracy or general superiority over other surrogate methods.
