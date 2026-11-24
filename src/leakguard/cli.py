"""CLI for leakguard."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from leakguard.auditor import audit_split, load_csv_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leakguard",
        description="Scientific ML train/test leakage auditor (duplicates, group, temporal, spatial, target proxy).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit", help="Audit train.csv and test.csv for data leakage")
    p.add_argument("train_csv")
    p.add_argument("test_csv")
    p.add_argument("--target", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--time", default=None)
    p.add_argument("--lat", default=None)
    p.add_argument("--lon", default=None)
    p.add_argument("--embargo-hours", type=float, default=0.0)
    p.add_argument("--spatial-buffer-km", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        rep = audit_split(
            load_csv_rows(args.train_csv),
            load_csv_rows(args.test_csv),
            target_col=args.target,
            group_col=args.group,
            time_col=args.time,
            lat_col=args.lat,
            lon_col=args.lon,
            embargo_hours=args.embargo_hours,
            min_spatial_buffer_km=args.spatial_buffer_km,
        )
        print(
            json.dumps(
                {
                    "passed": rep.passed,
                    "train_rows": rep.train_rows,
                    "test_rows": rep.test_rows,
                    "critical": rep.critical_count,
                    "high": rep.high_count,
                    "warnings": rep.warning_count,
                    "findings": [
                        {
                            "rule_id": f.rule_id,
                            "severity": f.severity.value,
                            "title": f.title,
                            "message": f.message,
                            "metrics": f.metrics,
                        }
                        for f in rep.findings
                    ],
                },
                indent=2,
            )
        )
        return 0 if rep.passed else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
