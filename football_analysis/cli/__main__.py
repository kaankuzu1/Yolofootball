"""Allows ``python -m football_analysis.cli``."""

import sys

from football_analysis.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
