"""Check that benchmark reproduction works outside the manuscript workspace."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != 'linux', reason='benchmark RSS collection uses Linux')
def test_standalone_benchmark_and_render(tmp_path):
    pytest.importorskip('qiskit')
    pytest.importorskip('matplotlib')
    root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / 'standalone'
    checkout.mkdir()
    shutil.copy(root / 'scripts_revision_benchmarks.py', checkout)
    # The installed package is used when this isolated directory has no src tree.
    output = tmp_path / 'results' / 'new'
    paper = tmp_path / 'manuscript'
    command = [sys.executable, str(checkout / 'scripts_revision_benchmarks.py'),
               '--output-dir', str(output), '--paper-dir', str(paper)]
    subprocess.run(command + ['--only', 'base', 'serial'], cwd=checkout,
                   check=True, capture_output=True, text=True)
    records = {name: (output / f'{name}.json').read_bytes()
               for name in ('base', 'serial')}
    for raw in records.values():
        row = json.loads(raw)['row']
        assert row['status'] == 'certified'
        assert row['error_bound'] <= 1e-10
    with (output / 'summary.csv').open() as stream:
        assert [row['name'] for row in csv.DictReader(stream)] == ['base', 'serial']
    assert (output / 'costs.png').stat().st_size > 0
    assert (output / 'validated_table.tex').read_bytes() == (paper / 'validated_table.tex').read_bytes()
    subprocess.run(command + ['--render'], cwd=checkout,
                   check=True, capture_output=True, text=True)
    assert records == {name: (output / f'{name}.json').read_bytes() for name in records}
