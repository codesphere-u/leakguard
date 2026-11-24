"""leakguard — Scientific ML dataset leakage auditor."""

from leakguard.auditor import (
    LeakageAuditReport,
    LeakFinding,
    LeakSeverity,
    audit_split,
    haversine_km,
    load_csv_rows,
)

__version__ = "0.1.0"

__all__ = [
    "LeakFinding",
    "LeakSeverity",
    "LeakageAuditReport",
    "__version__",
    "audit_split",
    "haversine_km",
    "load_csv_rows",
]
