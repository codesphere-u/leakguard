"""Deterministic tabular, time-series, and geospatial ML train/test leakage auditor."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

EARTH_RADIUS_KM = 6371.0088


class LeakSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    WARNING = "warning"


@dataclass(frozen=True)
class LeakFinding:
    rule_id: str
    severity: LeakSeverity
    title: str
    message: str
    remediation: str
    metrics: dict[str, float | int | str]


@dataclass(frozen=True)
class LeakageAuditReport:
    train_rows: int
    test_rows: int
    passed: bool
    critical_count: int
    high_count: int
    warning_count: int
    findings: tuple[LeakFinding, ...]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 5 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def _to_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        num = float(val)
        return num if math.isfinite(num) else None
    except (TypeError, ValueError):
        return None


def _parse_ts(val: Any) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(str(val).strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def audit_split(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    target_col: str | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    lat_col: str | None = None,
    lon_col: str | None = None,
    embargo_hours: float = 0.0,
    min_spatial_buffer_km: float = 5.0,
    proxy_corr_threshold: float = 0.95,
) -> LeakageAuditReport:
    """Audit train and test splits for 6 classes of scientific ML data leakage."""
    findings: list[LeakFinding] = []
    if not train_rows or not test_rows:
        return LeakageAuditReport(len(train_rows), len(test_rows), True, 0, 0, 0, ())

    all_cols = sorted(set(train_rows[0].keys()) & set(test_rows[0].keys()))
    feature_cols = [c for c in all_cols if c != target_col]

    # 1. LEAK-001: Exact duplicate feature rows across train and test
    def _row_hash(row: dict[str, Any]) -> str:
        payload = json.dumps(
            {c: str(row.get(c, "")).strip() for c in feature_cols},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    train_hashes = {_row_hash(r) for r in train_rows}
    dup_test_count = sum(1 for r in test_rows if _row_hash(r) in train_hashes)
    if dup_test_count > 0:
        dup_pct = round((dup_test_count / len(test_rows)) * 100.0, 2)
        findings.append(
            LeakFinding(
                rule_id="LEAK-001",
                severity=LeakSeverity.CRITICAL,
                title="Exact Duplicate Feature Rows Across Train & Test",
                message=f"{dup_test_count}/{len(test_rows)} test rows ({dup_pct}%) appear verbatim in train.",
                remediation="Deduplicate rows BEFORE splitting into train/validation/test sets.",
                metrics={"duplicate_test_rows": dup_test_count, "duplicate_pct": dup_pct},
            )
        )

    # 2. LEAK-002: Group-ID / Entity Contamination
    if group_col and group_col in all_cols:
        train_groups = {str(r.get(group_col, "")).strip() for r in train_rows if str(r.get(group_col, "")).strip()}
        test_groups = {str(r.get(group_col, "")).strip() for r in test_rows if str(r.get(group_col, "")).strip()}
        overlap_groups = train_groups & test_groups
        if overlap_groups:
            findings.append(
                LeakFinding(
                    rule_id="LEAK-002",
                    severity=LeakSeverity.CRITICAL,
                    title="Group-ID / Entity Contamination Across Splits",
                    message=(
                        f"{len(overlap_groups)} entities in '{group_col}' appear in both train and test "
                        f"(sample: {sorted(overlap_groups)[:3]})."
                    ),
                    remediation=f"Use GroupKFold / GroupShuffleSplit keyed on '{group_col}'.",
                    metrics={"overlapping_groups": len(overlap_groups), "group_col": group_col},
                )
            )

    # 3. LEAK-003: Temporal Look-Ahead & Embargo Violation
    if time_col and time_col in all_cols:
        train_times = [t for r in train_rows if (t := _parse_ts(r.get(time_col))) is not None]
        test_times = [t for r in test_rows if (t := _parse_ts(r.get(time_col))) is not None]
        if train_times and test_times:
            max_train = max(train_times)
            lookahead_count = sum(1 for t in test_times if t <= max_train)
            embargo_violations = sum(
                1 for t in test_times if 0.0 < (t - max_train).total_seconds() / 3600.0 < embargo_hours
            )
            if lookahead_count > 0:
                findings.append(
                    LeakFinding(
                        rule_id="LEAK-003",
                        severity=LeakSeverity.CRITICAL,
                        title="Temporal Look-Ahead Leakage",
                        message=(
                            f"{lookahead_count}/{len(test_times)} test rows occur on or before "
                            f"max(train.{time_col})={max_train.isoformat()}."
                        ),
                        remediation="Use strict chronological forward-chaining splits without shuffling.",
                        metrics={"lookahead_rows": lookahead_count, "max_train_time": max_train.isoformat()},
                    )
                )
            elif embargo_hours > 0.0 and embargo_violations > 0:
                findings.append(
                    LeakFinding(
                        rule_id="LEAK-003",
                        severity=LeakSeverity.HIGH,
                        title="Temporal Embargo Gap Violation",
                        message=(
                            f"{embargo_violations} test rows fall within the {embargo_hours}h "
                            "autocorrelation embargo window after train end."
                        ),
                        remediation="Enforce a purged embargo gap matching feature lookback/horizon length.",
                        metrics={"embargo_violations": embargo_violations, "embargo_hours": embargo_hours},
                    )
                )

    # 4. LEAK-004: Spatial Proximity Leakage (Haversine)
    if lat_col and lon_col and lat_col in all_cols and lon_col in all_cols:
        train_pts = [
            (lat, lon)
            for r in train_rows
            if (lat := _to_float(r.get(lat_col))) is not None
            and (lon := _to_float(r.get(lon_col))) is not None
        ]
        test_pts = [
            (lat, lon)
            for r in test_rows
            if (lat := _to_float(r.get(lat_col))) is not None
            and (lon := _to_float(r.get(lon_col))) is not None
        ]
        if train_pts and test_pts:
            near_count = 0
            min_dist_seen = float("inf")
            for t_lat, t_lon in test_pts:
                best = min(haversine_km(t_lat, t_lon, tr_lat, tr_lon) for tr_lat, tr_lon in train_pts)
                min_dist_seen = min(min_dist_seen, best)
                if best < min_spatial_buffer_km:
                    near_count += 1
            if near_count > 0:
                findings.append(
                    LeakFinding(
                        rule_id="LEAK-004",
                        severity=LeakSeverity.HIGH,
                        title="Spatial Autocorrelation Proximity Leakage",
                        message=(
                            f"{near_count}/{len(test_pts)} test locations lie within "
                            f"{min_spatial_buffer_km} km of a training point (closest={min_dist_seen:.2f} km)."
                        ),
                        remediation="Use spatial block cross-validation with buffer >= spatial decorrelation range.",
                        metrics={"near_test_points": near_count, "min_distance_km": round(min_dist_seen, 3)},
                    )
                )

    # 5. LEAK-005: Target Proxy Feature Correlation
    if target_col and target_col in all_cols:
        for col in feature_cols:
            if col in (group_col, time_col, lat_col, lon_col):
                continue
            pairs = [
                (x, y)
                for r in train_rows
                if (x := _to_float(r.get(col))) is not None and (y := _to_float(r.get(target_col))) is not None
            ]
            if len(pairs) >= 5:
                r_val = _pearson_r([p[0] for p in pairs], [p[1] for p in pairs])
                if r_val is not None and abs(r_val) >= proxy_corr_threshold:
                    findings.append(
                        LeakFinding(
                            rule_id="LEAK-005",
                            severity=LeakSeverity.HIGH,
                            title="Suspicious Target-Proxy Feature Leakage",
                            message=(
                                f"Feature '{col}' has |r|={abs(r_val):.4f} with target '{target_col}' "
                                f"(threshold={proxy_corr_threshold})."
                            ),
                            remediation="Verify whether this column is computed post-outcome or encodes the label.",
                            metrics={"feature": col, "pearson_r": round(r_val, 5)},
                        )
                    )

    # 6. LEAK-006: Train/Test Distribution Scale Shift (Mixed SI/Imperial Unit Hazard)
    for col in feature_cols:
        if col in (group_col, time_col, lat_col, lon_col):
            continue
        tr_vals = [v for r in train_rows if (v := _to_float(r.get(col))) is not None]
        te_vals = [v for r in test_rows if (v := _to_float(r.get(col))) is not None]
        if len(tr_vals) >= 5 and len(te_vals) >= 5:
            tr_med = sorted(tr_vals)[len(tr_vals) // 2]
            te_med = sorted(te_vals)[len(te_vals) // 2]
            if abs(tr_med) > 1e-6 and abs(te_med) > 1e-6:
                ratio = max(abs(tr_med), abs(te_med)) / min(abs(tr_med), abs(te_med))
                if ratio >= 10.0:
                    findings.append(
                        LeakFinding(
                            rule_id="LEAK-006",
                            severity=LeakSeverity.WARNING,
                            title="Order-of-Magnitude Unit/Scale Mismatch Between Splits",
                            message=(
                                f"Column '{col}' median shifts by {ratio:.1f}x "
                                f"(train={tr_med:.3g}, test={te_med:.3g}); check SI vs imperial units."
                            ),
                            remediation="Standardize physical units across all data sources before modeling.",
                            metrics={"column": col, "median_ratio": round(ratio, 2)},
                        )
                    )

    crit = sum(1 for f in findings if f.severity == LeakSeverity.CRITICAL)
    high = sum(1 for f in findings if f.severity == LeakSeverity.HIGH)
    warn = sum(1 for f in findings if f.severity == LeakSeverity.WARNING)
    return LeakageAuditReport(
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        passed=(crit == 0 and high == 0),
        critical_count=crit,
        high_count=high,
        warning_count=warn,
        findings=tuple(findings),
    )
