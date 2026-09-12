"""Command-line interface for validated deployed-instance surrogate construction."""
import argparse
import json
from dataclasses import asdict
from .certifier import EXAMPLES, certify, print_report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names',nargs='*')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--tolerance',type=float,default=1e-10)
    parser.add_argument('--dps',type=int,default=40)
    parser.add_argument('--max-basis',type=int,default=8000)
    parser.add_argument('--diagnostics',action='store_true')
    args=parser.parse_args(argv)
    if args.list:
        print('\n'.join(EXAMPLES));return 0
    if args.selftest:
        import numpy as np
        from .validated import evaluate_surrogate
        from .certifier import simulate
        spec=EXAMPLES['hqz_base']();rep=certify(spec)
        x=np.random.RandomState(11).uniform(-2,8,(50,4))
        exact=simulate(spec,x,rep.deployed_params)
        assert rep.verdict.startswith('CERTIFIED')
        assert max(np.max(np.abs(evaluate_surrogate(b,x)-exact[b['readout']])) for b in rep.basis)<1e-12
        print('Base certificate and numerical reference check passed.');return 0
    for name in args.names or ['hqz_base']:
        if name not in EXAMPLES:
            parser.error(f'unknown encoder: {name}')
        rep=certify(EXAMPLES[name](),tolerance=args.tolerance,dps=args.dps,max_basis=args.max_basis,diagnostics=args.diagnostics)
        if args.json:
            print(json.dumps(asdict(rep)))
        else:
            print_report(rep)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
