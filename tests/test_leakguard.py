"""Tests for leakguard scientific ML leakage checks and CLI."""

from __future__ import annotations

import csv
from pathlib import Path

from leakguard import audit_split, haversine_km
from leakguard.cli import main as cli_main


def test_clean_chronological_spatially_separated_split_passes() -> None:
    train = [
        {
            "patient_id": f"P{i}",
            "ts": f"2026-01-0{i + 1}T00:00:00Z",
            "lat": 0.31,
            "lon": 32.58,
            "temp_c": 36.5 + 0.1 * i,
            "label": i % 2,
        }
        for i in range(6)
    ]
    test = [
        {
            "patient_id": f"P{i + 20}",
            "ts": f"2026-02-0{i + 1}T00:00:00Z",
            "lat": 2.77,
            "lon": 32.29,
            "temp_c": 36.8 + 0.1 * i,
            "label": i % 2,
        }
        for i in range(6)
    ]
    rep = audit_split(
        train,
        test,
        target_col="label",
        group_col="patient_id",
        time_col="ts",
        lat_col="lat",
        lon_col="lon",
        embargo_hours=24.0,
        min_spatial_buffer_km=10.0,
    )
    assert rep.passed is True
    assert rep.findings == ()
    assert haversine_km(0.31, 32.58, 2.77, 32.29) > 250.0


def test_catches_all_six_leakage_classes() -> None:
    train = [
        {
            "pid": f"G{i}",
            "ts": f"2026-05-1{i}T00:00:00Z",
            "lat": 0.310,
            "lon": 32.580,
            "dose": 10.0 + i,
            "proxy": 2.0 * (i + 1),
            "y": float(i + 1),
        }
        for i in range(6)
    ]
    test = [
        {
            "pid": "G0",
            "ts": "2026-05-10T00:00:00Z",
            "lat": 0.310,
            "lon": 32.580,
            "dose": 10.0,
            "proxy": 2.0,
            "y": 1.0,
        },
        *[
            {
                "pid": f"T{i}",
                "ts": "2026-05-11T00:00:00Z",
                "lat": 0.311,
                "lon": 32.581,
                "dose": 1500.0 + i,
                "proxy": 2.0 * (i + 1),
                "y": float(i + 1),
            }
            for i in range(5)
        ],
    ]
    rep = audit_split(
        train,
        test,
        target_col="y",
        group_col="pid",
        time_col="ts",
        lat_col="lat",
        lon_col="lon",
        min_spatial_buffer_km=5.0,
    )
    assert rep.passed is False
    rules = {f.rule_id for f in rep.findings}
    assert {"LEAK-001", "LEAK-002", "LEAK-003", "LEAK-004", "LEAK-005", "LEAK-006"}.issubset(rules)


def test_cli_csv_audit(tmp_path: Path) -> None:
    tr = tmp_path / "train.csv"
    te = tmp_path / "test.csv"
    with tr.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["gid", "x", "y"])
        w.writeheader()
        w.writerow({"gid": "A", "x": "1", "y": "0"})
    with te.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["gid", "x", "y"])
        w.writeheader()
        w.writerow({"gid": "A", "x": "1", "y": "0"})
    assert cli_main(["audit", str(tr), str(te), "--group", "gid", "--target", "y"]) == 2
