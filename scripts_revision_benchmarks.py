"""Generate certificate records, measured costs and optional manuscript tables.

Each case runs in a fresh subprocess so peak RSS is the total worker peak, not a
cumulative peak from previous cases. Timings exclude imports but include planning,
interval construction, pruning and reporting. Holdout comparisons are separate.
"""
import dataclasses
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))
import numpy as np
from deqcert import EXAMPLES, certify, evaluate_surrogate, simulate

NAMES=['base','cross','reupload','vqc8','zz3','zz4linear','zz4full','z4','zyy3','serial']
LABELS={'base':'Base pair encoder','cross':'Cross-pair encoder','reupload':'Re-uploading pair (fixed instance)',
        'vqc8':'Eight-qubit VQC','zz3':'ZZ map, $n=3$, two layers',
        'zz4linear':'Library ZZ (linear) + RealAmplitudes',
        'zz4full':'Library ZZ (full) + RealAmplitudes',
        'z4':'Library Z (two layers) + EfficientSU2',
        'zyy3':'Library Pauli (Z, YY) + RealAmplitudes','serial':'Serial re-uploading, $L=3$'}


def make(name):
    own={'base':'hqz_base','cross':'hqz_cross','reupload':'reenc_pair','vqc8':'vqc_facedet','zz3':'zz_feature_map'}
    if name in own:
        return EXAMPLES[own[name]]()
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.circuit.library import ZZFeatureMap,ZFeatureMap,PauliFeatureMap,RealAmplitudes,EfficientSU2
    from deqcert import from_qiskit
    if name=='serial':
        x=Parameter('x'); th=ParameterVector('th',6);qc=QuantumCircuit(1)
        for i in range(3):
            qc.rx(x,0);qc.ry(th[2*i],0);qc.rz(th[2*i+1],0)
        return from_qiskit(qc,[x],['Z0'],name)
    fm,ans={
        'zz4linear':lambda:(ZZFeatureMap(4,reps=1,entanglement='linear'),RealAmplitudes(4,reps=1)),
        'zz4full':lambda:(ZZFeatureMap(4,reps=1,entanglement='full'),RealAmplitudes(4,reps=1)),
        'z4':lambda:(ZFeatureMap(4,reps=2),EfficientSU2(4,reps=1)),
        'zyy3':lambda:(PauliFeatureMap(3,paulis=['Z','YY'],reps=1),RealAmplitudes(3,reps=1)),
    }[name]()
    qc=fm.compose(ans).decompose()
    return from_qiskit(qc,list(fm.parameters),[f'Z{q}' for q in range(qc.num_qubits)],name)


def worker(name):
    import mpmath, qiskit
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
              if line.startswith('model name')),platform.processor())
    spec=make(name)
    t=time.perf_counter();rep=certify(spec,seed=0);elapsed=time.perf_counter()-t
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    empirical=None
    if rep.basis:
        x=np.random.RandomState(719).uniform(-2,8,(100,spec.n_features))
        truth=simulate(spec,x,rep.deployed_params)
        empirical=max(float(np.max(np.abs(evaluate_surrogate(b,x)-truth[b['readout']]))) for b in rep.basis)
    row=dict(name=name,n=spec.n_qubits,readouts=len(spec.readouts),
             grids='/'.join(str(b) for b in rep.cost['candidate_grid_points']),
             queries=rep.cost['construction_queries'],modes=rep.eff_dim,error_bound=rep.certificate_error,
             time_s=elapsed,peak_rss_mb=peak,empirical_error=empirical,
             status=('certified' if rep.verdict.startswith('CERTIFIED') else
                     'tolerance not met' if rep.verdict.startswith('SURROGATE FOUND') else 'inconclusive'))
    return dict(row=row,report=rep.__dict__,specification=dataclasses.asdict(spec),metadata=dict(python=platform.python_version(),
                numpy=np.__version__,mpmath=mpmath.__version__,qiskit=qiskit.__version__,
                platform=platform.platform(),cpu=cpu,precision_dps=40,seed=0,tolerance=1e-10,prune=1e-14,
                max_basis=8000,max_active_qubits=6,max_work=20_000_000,max_harmonic=32))


def sci(v):
    return '---' if v is None else r'$'+f'{v:.2g}'.replace('e-',r'\times10^{-').replace('e+',r'\times10^{')+('}' if 'e' in f'{v:.2g}' else '')+'$'


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=NAMES);p.add_argument('--only',nargs='+',choices=NAMES);p.add_argument('--render',action='store_true')
    p.add_argument('--output-dir',type=Path,default=HERE/'benchmark_results',
                   help='directory for JSON records, CSV, LaTeX rows and plot')
    p.add_argument('--paper-dir',type=Path,
                   help='also write validated_table.tex to this manuscript directory')
    args=p.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker)));return
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    for name in ([] if args.render else (args.only or NAMES)):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        raw=subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'--worker',name],env=env,text=True)
        result=json.loads(raw)
        (out/f'{name}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result['row']),flush=True)
    rows=[json.loads((out/f'{n}.json').read_text())['row']
          for n in NAMES if (out/f'{n}.json').exists()]
    if not rows:
        p.error(f'no benchmark records found in {out}')
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=[]
    for r in rows:
        status='budget' if r['status']=='inconclusive' else r['status']
        lines.append(f"{LABELS[r['name']]} & {r['n']} & {r['queries']:,} & {r['modes'] if r['modes'] is not None else '---'} & {sci(r['error_bound'])} & {r['time_s']:.3f} & {r['peak_rss_mb']:.1f} & {status} \\\\")
    table='\\newcommand{\\validatedrows}{%\n'+'\n'.join(lines)+'}\n'
    (out/'validated_table.tex').write_text(table)
    if args.paper_dir:
        args.paper_dir.mkdir(parents=True,exist_ok=True)
        (args.paper_dir/'validated_table.tex').write_text(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    good=[r for r in rows if r['status']=='certified']
    fig,axes=plt.subplots(1,3,figsize=(10,3.1))
    for r in good:
        q=r['queries'];axes[0].scatter(q,r['time_s']);axes[1].scatter(q,r['peak_rss_mb']);axes[2].scatter(q,r['error_bound'])
        for panel,(ax,field) in enumerate(zip(axes,['time_s','peak_rss_mb','error_bound'])):
            dx,dy={(0,'serial'):(4,11),(0,'base'):(4,0),
                   (0,'zz4linear'):(-4,9),(0,'zz3'):(4,-12),
                   (0,'z4'):(-4,6),(0,'zyy3'):(4,-12),
                   (1,'z4'):(-4,5),(1,'zz4linear'):(-4,6),
                   (2,'zz4linear'):(-4,-10)}.get((panel,r['name']),(4,4))
            ax.annotate(r['name'],(q,r[field]),fontsize=7,xytext=(dx,dy),
                        ha='right' if dx<0 else 'left',textcoords='offset points')
    for ax in axes:
        ax.set_xscale('log');ax.set_xlim(4,40000);ax.set_xlabel('Interval circuit evaluations');ax.grid(alpha=.2)
    axes[0].set_yscale('log');axes[0].set_ylabel('Construction wall time (s)')
    axes[1].set_ylabel('Worker peak RSS (MiB)')
    axes[2].set_yscale('log');axes[2].set_ylabel('Global error bound')
    axes[0].set_ylim(.006,100);axes[1].set_ylim(65,111);axes[2].set_ylim(.9e-16,7e-16)
    fig.tight_layout();fig.savefig(out/'costs.png',dpi=220);plt.close(fig)


if __name__=='__main__':
    main()
