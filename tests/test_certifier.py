"""Regression tests for the deqcert certifier.

Runs the full pipeline on the four encoders reported in the paper and asserts
the expected verdicts.

  hqz_base     -> dequantizable, exact surrogate (residual ~ machine precision)
  hqz_cross    -> dequantizable despite a large full DLA (readout class bounded)
  reenc_pair   -> frequency escape (unbounded encoding frequencies)
  vqc_facedet  -> algebra escape (exponential DLA, entangled)  [slow, ~100 s]

Run:  pytest                     everything (~100 s, dominated by vqc_facedet)
      pytest -m "not slow"       the fast subset (~1 s)
"""
import pytest

from deqcert import EXAMPLES, certify


def _certify(name):
    return certify(EXAMPLES[name]())


@pytest.mark.parametrize("name", ["hqz_base", "hqz_cross"])
def test_dequantizable_with_exact_surrogate(name):
    r = _certify(name)
    assert "DEQUANTIZABLE" in r.verdict
    assert r.surrogate_residual is not None
    assert r.surrogate_residual < 1e-10


def test_reenc_pair_is_a_frequency_escape():
    r = _certify("reenc_pair")
    assert "ESCAPED via the encoding-frequency" in r.verdict
    assert r.freq_class == "unbounded"


@pytest.mark.slow
def test_vqc_facedet_is_an_algebra_escape():
    r = _certify("vqc_facedet")
    assert "ESCAPED via the dynamical Lie algebra" in r.verdict
    assert r.dla_class.startswith("exponential")


def test_hqz_base_surrogate_matches_paper():
    """Table VI of the paper: B = 8 terms, residual at machine precision."""
    r = _certify("hqz_base")
    assert r.eff_dim == 8
    assert r.surrogate_residual < 1e-12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
