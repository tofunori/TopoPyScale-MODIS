"""Shared argparse helpers for ERA5 CLI entry points."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_region_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="RGI region code (e.g. 02-03)",
    )


def add_db_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to DuckDB database",
    )


def parse_months(spec: str) -> list[int]:
    """Parse comma-separated month numbers: '06,07,08' -> [6, 7, 8]."""
    return [int(m.strip()) for m in spec.split(",")]
