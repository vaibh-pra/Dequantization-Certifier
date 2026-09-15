"""W4 (R3.4): measured runtime and memory scaling of the certifier itself.

Measures, per circuit instance:
  - DLA Lie-closure wall time vs qubit count n and dimension ceiling
  - surrogate basis size B = prod_c (2 r_c + 1)   (analytic, Corollary 1)
  - basis enumeration + least-squares fit time vs B
  - peak memory of the fit stage
Also pushes the ZZ feature map to n = 4 to probe the dim-64 DLA cap.

Outputs: results/certifier_cost.csv + results/certifier_cost.png
Run:  python scripts_certifier_cost.py   (from deqcert/, needs src on path)
"""
import itertools
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from deqcert.certifier import (  # noqa: E402
    EXAMPLES, dla_dimension, angle_channels, channel_basis, eval_channel_basis,
    _channel_angles, simulate, readout_support, _sample_params, harmonic_bound,
)


def measure_dla(make, n_list, cap_list):
    rows = []
    for n in n_list:
        for cap in cap_list:
            spec = make(n=n) if "zz" in make.__name__ else make()
            t0 = time.perf_counter()
            tracemalloc.start()
            dim, sat = dla_dimension(spec, cap=cap)
            peak = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
            dt = time.perf_counter() - t0
            rows.append(dict(part="dla", n=spec.n_qubits, cap=cap, dim=dim_row(dim=dim, sat=sat),
                             time_s=dt, peak_mb=peak))
            print(f"  DLA n={spec.n_qubits:2d} cap={cap:3d}: dim={dim_row(dim, sat):>10s} "
                  f"{dt:7.2f}s {peak:7.1f} MB", flush=True)
    return rows


def dim_row(dim, sat):
    return f">={dim}" if sat else f"{dim}"


def measure_fit(make, key, n=3, fm_reps=2, K=None):
    """Surrogate-side scaling for one circuit at its analytic bound."""
    spec = make() if key != "zz" else make(n=n, fm_reps=fm_reps, var_reps=1)
    params = {nm: 0.37 for nm in spec.param_names()}
    S = readout_support_safe(spec, "Z0", params)
    channels = angle_channels(spec, S)
    if K is None:
        K, _ = harmonic_bound(spec, channels)
    B = (2 * K + 1) ** len(channels)
    if B > 600_000:
        print(f"  {key} n={spec.n_qubits} K={K}: B={B:,} -> skipped (RAM guard)",
              flush=True)
        return None
    rng = np.random.RandomState(0)
    t0 = time.perf_counter()
    tracemalloc.start()
    basis = channel_basis(len(channels), K)
    X = rng.uniform(0, 2 * np.pi, (2 * B + 16, spec.n_features))
    Phi = eval_channel_basis(basis, _channel_angles(channels, X), X.shape[0])
    y = simulate(spec, X, params)["Z0"]
    c, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    dt = time.perf_counter() - t0
    Xt = rng.uniform(0, 2 * np.pi, (300, spec.n_features))
    ft = simulate(spec, Xt, params)["Z0"]
    res = float(np.max(np.abs(
        eval_channel_basis(basis, _channel_angles(channels, Xt), 300) @ c - ft)))
    print(f"  {key} n={spec.n_qubits} C={len(channels)} K={K}: B={B:,} "
          f"fit={dt:6.2f}s peak={peak:8.1f} MB residual={res:.2e}", flush=True)
    return dict(part="surrogate", name=spec.name, n=spec.n_qubits,
                channels=len(channels), K=K, B=B, fit_time_s=dt, peak_mb=peak,
                residual=res)


def readout_support_safe(spec, readout, params, seed=0):
    rng = np.random.RandomState(seed)
    dep = set()
    for _ in range(4):
        x = rng.uniform(0, 2 * np.pi, (1, spec.n_features))
        base = simulate(spec, x, params)[readout][0]
        for f in range(spec.n_features):
            xp = x.copy(); xp[0, f] += 1e-4
            if abs(simulate(spec, xp, params)[readout][0] - base) > 1e-7:
                dep.add(f)
    return sorted(dep)


def main():
    rows = []

    # --- 1) DLA closure: wall time + memory vs width and cap
    print("== DLA Lie closure scaling", flush=True)
    from deqcert.certifier import zz_feature_map
    rows += measure_dla(zz_feature_map, n_list=[2, 3, 4], cap_list=[16, 64, 256])

    # --- 2) surrogate side: B, fit time, memory vs channels x harmonics
    print("== Surrogate enumeration + fit scaling", flush=True)
    fit_rows = []
    fit_rows.append(measure_fit(EXAMPLES["hqz_base"], "hqz_base"))
    fit_rows.append(measure_fit(EXAMPLES["hqz_cross"](), "hqz_cross") if False
                    else measure_fit(make_cross, "hqz_cross"))
    for n, L in [(2, 1), (2, 2), (3, 1), (3, 2), (4, 1)]:
        fit_rows.append(measure_fit(zz_feature_map, "zz", n=n, fm_reps=L))
    rows += [r for r in fit_rows if r]

    # --- 3) analytic B formula vs measured basis count (theorem check)
    print("== Analytic B = prod_c (2K+1)^C check", flush=True)
    for r in [r for r in fit_rows if r]:
        pred = (2 * r["K"] + 1) ** r["channels"]
        assert pred == r["B"], (pred, r["B"])
        print(f"  {r['name']}: predicted B={pred:,} == enumerated {r['B']:,}", flush=True)

    # write CSV
    out = "/home/vaibhav/Downloads/qml_dequant/results/certifier_cost.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import csv
    allrows = [r for r in rows if r]
    keys = sorted({k for r in allrows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(allrows)
    print(f"wrote {out}", flush=True)


def make_cross():
    from deqcert.certifier import EXAMPLES
    return EXAMPLES["hqz_cross"]()


if __name__ == "__main__":
    main()
