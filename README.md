> ⚠️ **Archived as of 2026-06-27** — superseded in part by [danstonedev/imu](https://github.com/danstonedev/imu); retained read-only for history.
>
> **Salvage note:** `py/hip_inverse_dynamics.py` holds a unique 3-segment (foot→shank→thigh) Newton-Euler inverse-dynamics chain with De Leva anthropometrics, force-plate/insole CoP, and an IMU-only GRF/CoP estimator — found nowhere else in the org (the `imu` successor has only a simpler femur-only model). Salvage `py/hip_inverse_dynamics.py`, `py/pages_pipeline.py`, and `js/gpu/` before relying on the archive. Evidence: [devpt/LEGACY-AUDIT.md](https://github.com/danstonedev/devpt/blob/claude/devpt-portfolio-analysis-whhyrm/LEGACY-AUDIT.md).

# IMU Hip Torque (Browser MVP)

## GitHub Pages
You can host this app on GitHub Pages. Steps:

1. In your repo settings, enable Pages and select the `main` branch with root (`/`).
2. Ensure `.nojekyll` exists at the repo root (present).
3. Wait for Pages to build, then open the provided URL.

Notes:
- The app is static and runs entirely client-side using Pyodide and Chart.js.
- Large CSVs run in-browser; performance depends on your machine.
