"""Compatibility driver for the revised validated benchmark suite."""
import sys
from scripts_revision_benchmarks import main
if __name__=="__main__":
    sys.argv=[sys.argv[0]]+[]+sys.argv[1:]
    main()
