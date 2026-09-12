# deqcert 0.2.0

Construct explicit Fourier surrogates for **fixed deployed quantum encoder instances**, with a uniform absolute error bound. The certificate refers to the normalized gate specification; it is not a quantum-advantage decision or a guarantee of efficient simulation at arbitrary width.

```bash
git clone https://github.com/vaibh-pra/Dequantization-Certifier.git
cd Dequantization-Certifier
pip install -e '.[qiskit,test]'
python -m deqcert hqz_base
pytest
```

```python
from deqcert import certify, EXAMPLES, evaluate_surrogate, evaluate_enclosure

spec = EXAMPLES['hqz_base']()
params = {name: 0.37 for name in spec.param_names()}
report = certify(spec, deployed_params=params, tolerance=1e-10)
print(report.verdict, report.certificate_error)
x = [[0.1, 0.2, 0.3, 0.4]]
fast = evaluate_surrogate(report.basis[0], x)
interval = evaluate_enclosure(report.basis[0], x)
```

`CERTIFIED epsilon surrogate` means that the returned mathematical model has uniform error at most the reported bound, which is at most the requested tolerance. `SURROGATE FOUND` means construction completed but did not meet that tolerance. `INCONCLUSIVE (resource budget)` means the complete call was declined before circuit evaluation. Invalid specifications raise errors. No outcome proves quantum hardness.

The supported gates are RX/RY/RZ, controlled rotations about those axes, CNOT and H. Initial states are all-zero product states and readouts are products of Z operators. Data angles are scaled features or finite polynomials; other angles are fixed or deployed parameters. Trainable scales are frozen at their recorded values. Omitting `deployed_params` selects a reproducible demonstration instance; this is explicitly identified in the report.

The Qiskit importer uses symbolic polynomial parsing and rejects unsupported expressions, measurements and classical control. Numeric constants are normalized to binary64. A certificate concerns this normalized `EncoderSpec`, not unrounded source expressions or physical-device noise.

Direct specifications must likewise supply constants and deployed values exactly representable in binary64; values requiring conversion are rejected until normalized explicitly. The serialized field `eff_dim` is retained for compatibility but now counts retained complex Fourier modes, not a function-class dimension; prefer `report.cost['retained_fourier_modes']`.

The algorithm retains a conservative reverse causal cone, replaces encoding expressions by independent channel variables, and reconstructs their complete tensor-grid Fourier representation. Elementary mpmath interval operations enclose the circuit evaluations and DFT coefficients. The sum of coefficient errors bounds the function error uniformly because the complex phase factors have modulus one. Pruned coefficients contribute their full enclosures. Channel dependencies on the original input domain do not require a sampling-rank assumption.

The guarantee covers the mathematical model with returned binary64 coefficients treated exactly. `evaluate_surrogate` is a fast floating-point evaluator with additional rounding error. `evaluate_enclosure` uses interval arithmetic and returns outward-rounded pointwise intervals including both model-construction and evaluation errors. These guarantees rely on the specified elementary interval operations; the mpmath documentation describes its interval context as experimental. The code is tested but is not a formally verified implementation.

Defaults: 40 decimal digits, tolerance `1e-10`, pruning threshold `1e-14`, at most 8,000 grid points per readout, at most six active qubits, channel order at most 32, and work estimate at most 20,000,000. Budget counts are computed before grid allocation or simulation. Optional DLA and sample-rank diagnostics are disabled by default and never affect the certificate. The legacy `extract_surrogate` least-squares helper is exploratory and always returns `certified=False`; manual `K` overrides cannot obtain a certificate through `certify`.

The construction needs gate-level access to evaluate independent channel combinations, not just black-box evaluations on the original input domain. Its grid can grow exponentially. Successful certification does not imply sparse optimality or faster execution than a statevector.

Clone this repository and run the certificate benchmark on Linux:

```bash
pip install -e '.[benchmark]'
python scripts_revision_benchmarks.py --only base serial
python scripts_revision_benchmarks.py
```

Results go to `benchmark_results/` by default, including JSON records, a CSV summary, LaTeX table rows and a plot. `--only` selects cases and produces a summary for available records. Each certificate records deployed parameters, modes, bounds, query counts, versions and its normalized specification. `--render` rebuilds the table and plot without repeating construction. `--output-dir` selects another results directory, and `--paper-dir` optionally copies the table macro into a manuscript directory. No surrounding manuscript checkout is needed. Timing uses Linux process peak RSS and fresh worker processes.

The recorded ten-case results accompanying the manuscript are in [examples/validated_revision](examples/validated_revision/README.md). Eight cases meet the requested tolerance and two are declined by the resource policy. These records report individual measurements, not a scaling law or a runtime advantage. To reproduce the manuscript outputs from a checkout placed inside its source workspace, use:

~~~bash
python scripts_revision_benchmarks.py \
  --output-dir ../results/validated_revision --paper-dir ../paper
~~~

The image-reconstruction audit belongs to the separate manuscript workspace and requires its saved checkpoints, local LFW images and experiment code. Those files are not dependencies of this package and are not included in this repository.

The 0.2.0 Fourier-mode counts differ from the historical real-product least-squares term counts. Historical CSVs and revision backups are not outputs of this implementation.

See [CHANGELOG.md](CHANGELOG.md) for migration details and [docs/related_work.md](docs/related_work.md) for the relationship to existing Fourier-surrogate and interval-transform methods.
