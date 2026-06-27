> ⚠️ **Archived as of 2026-06-27** — retained read-only for history. Its IMU/biomech lineage is superseded by [danstonedev/MASH](https://github.com/danstonedev/MASH) (TypeScript); note that `danstonedev/imu` is also archived, so it is not a live successor.
>
> **Salvage note:** this repo holds a unique 3-segment (foot→shank→thigh) Newton-Euler inverse-dynamics chain with De Leva anthropometrics, force-plate/insole CoP, and an IMU-only GRF/CoP estimator (`py/hip_inverse_dynamics.py`, `py/pages_pipeline.py`, `js/gpu/`) — found nowhere else in the org and preserved in this repo's git history. Copy it out before reuse. Evidence: [devpt/LEGACY-AUDIT.md](https://github.com/danstonedev/devpt/blob/claude/devpt-portfolio-analysis-whhyrm/LEGACY-AUDIT.md).

# IMU Hip Torque (Browser MVP)

## GitHub Pages
You can host this app on GitHub Pages. Steps:

1. In your repo settings, enable Pages and select the `main` branch with root (`/`).
2. Ensure `.nojekyll` exists at the repo root (present).
3. Wait for Pages to build, then open the provided URL.

Notes:
- The app is static and runs entirely client-side using Pyodide and Chart.js.
- Large CSVs run in-browser; performance depends on your machine.
