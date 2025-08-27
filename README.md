# IMU-only Hip Torque (MVP) — GitHub Pages App

This repository hosts a **static web app** (GitHub Pages) that computes **sagittal-plane hip torque** from **5 IMUs** (pelvis, L/R thigh, L/R tibia), with **no servers** and **no extra hardware**.

- ✅ **Strict overlap cleaning** — trims to the common time window across all sensors
- ✅ **Standing calibration** — uses quiet standing at the beginning/end to correct sensor-to-segment offsets
- ✅ **IMU-only GRF/CoP** — vertical GRF from pelvis acceleration + heel→toe rocker CoP
- ✅ **Inverse dynamics** — Newton–Euler 3-link (foot→shank→thigh)
- ✅ **Fully client-side** — your data never leaves the browser (via Pyodide)

## Quick start (GitHub Pages)

1. Create a new repo and copy this folder’s contents into it.
2. Commit and push.
3. In your repo, open **Settings → Pages** and set:
   - **Source:** `main` (or your default) branch
   - **Folder:** `/ (root)`
4. The site will be available at `https://<your-username>.github.io/<repo-name>/`.

## Using the app

1. Open the page. Upload the five CSVs (XSENS Dots exports).
2. Enter **Height (m)** and **Mass (kg)** (e.g., 1.70 m, 95.25 kg).
3. Keep **Standing calibration** and **Strict overlap cleaning** checked (recommended).
4. Click **Run analysis**.
5. View the **time-series** and **cycle-normalized** comparison plots.
6. Download **left/right CSVs**.

## Data expectations

- CSVs contain headers including `Quat_W, Quat_X, Quat_Y, Quat_Z`, and `FreeAcc_X/Y/Z`.  
- If present, `SampleTimeFine` (10 kHz ticks) is used to build real timestamps. Otherwise, 60 Hz is assumed.
- Files should correspond to:
  - Pelvis, Left Thigh, Right Thigh, Left Tibia, Right Tibia (one side each).

## Notes (MVP constraints)

- Pelvis linear acceleration (Xsens **FreeAcc** in world) is used as a proxy for whole-body COM acceleration → **vertical GRF** only.
- **Foot orientation ≈ tibia**, via a simple rocker trajectory; **pelvis height** is fixed (scaled to subject stature).
- Inertial properties via **de Leva** segment approximations.
- These outputs are **teaching/feedback grade** rather than publication-grade. Add force hardware for research.

## Local preview

Just open `index.html` in a modern browser. The app loads **Pyodide** from CDN and runs entirely locally.

## Credits

- Inverse dynamics and pipeline by ChatGPT (GPT-5 Thinking), 2025.
- MIT Licensed.
