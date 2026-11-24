# Architecture — `leakguard`

- `leakguard.auditor`: Pure-Python row-hash deduplication, group overlap set math, ISO-8601 chronological + embargo checks, WGS84 Haversine proximity scan, Pearson target-proxy correlation, and median-ratio unit-shift diagnostics.
- `leakguard.cli`: CI exit-code CLI (`0` clean, `2` critical/high leakage detected).
