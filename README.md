# leakguard

**Deterministic scientific ML train/test leakage auditor: exact duplicates, group-ID contamination, temporal look-ahead & embargo violations, Haversine spatial proximity leakage, target-proxy correlation, and SI/imperial scale anomalies.**

Zero runtime dependencies. Pure Python 3.11+. Designed to prevent the reproducibility failures documented in Kapoor & Narayanan (*"Leakage and the Reproducibility Crisis in Machine-Learning-Based Science"*) across healthcare, climate, agritech, epidemiology, and financial ML pipelines.

## Detectors

- `LEAK-001` — Exact duplicate feature rows across train & test splits (canonical SHA-256).
- `LEAK-002` — Group-ID / entity contamination (patient, farm, household, user ID overlap).
- `LEAK-003` — Temporal look-ahead & autocorrelation embargo violations.
- `LEAK-004` — Spatial autocorrelation proximity leakage (great-circle Haversine buffer in km).
- `LEAK-005` — Suspicious target-proxy feature correlation (`|r| >= 0.95`).
- `LEAK-006` — Order-of-magnitude train/test unit/scale mismatch warnings.

## Quick Start

```bash
uv sync --extra dev
leakguard audit train.csv test.csv --target outcome --group patient_id --time recorded_at --lat lat --lon lon --embargo-hours 24
```

## License

MIT License — see [LICENSE](LICENSE).
