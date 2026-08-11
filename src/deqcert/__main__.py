"""Command-line interface for deqcert.

    deqcert                 certify every built-in encoder
    deqcert hqz_base        certify a named built-in encoder
    deqcert --selftest      run the internal validation phases
    deqcert --list          list the built-in encoders
"""
from __future__ import annotations

import sys

from .certifier import EXAMPLES, certify, print_report


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    if args and args[0] == "--list":
        for nm in EXAMPLES:
            print(nm)
        return 0

    if args and args[0] == "--selftest":
        from .certifier import _p0_smoke, _p1_test, _p2p3_test, _p4_test
        _p0_smoke()
        _p1_test()
        _p2p3_test()
        _p4_test()
        return 0

    names = args if args else list(EXAMPLES)
    status = 0
    for nm in names:
        if nm not in EXAMPLES:
            print(f"unknown encoder '{nm}'. available: {list(EXAMPLES)}", file=sys.stderr)
            status = 2
            continue
        print_report(certify(EXAMPLES[nm]()))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
