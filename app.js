import { initWebGPU, rotateVecsGPU } from './js/gpu/quats.webgpu.js';
let pyodide = null;
let tsChart = null, cycleChart = null;
let leftURL = null, rightURL = null;
const ASSET_VERSION = 'v2025-08-27-4';

async function loadPyodideAndPackages() {
  if (pyodide) return pyodide;
  document.getElementById('status').textContent = 'Loading Python (Pyodide)...';
  console.time('pyodide:init');
  pyodide = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/' });
  console.timeEnd('pyodide:init');
  // Load required Python packages from the Pyodide distribution
  document.getElementById('status').textContent = 'Loading Python packages (numpy, pandas)...';
  try {
    console.time('pyodide:packages');
    await pyodide.loadPackage([ 'numpy', 'pandas' ]);
    console.timeEnd('pyodide:packages');
  } catch (e) {
    console.error('Failed to load Pyodide packages', e);
    document.getElementById('status').textContent = 'Error loading Python packages: ' + e.message;
    throw e;
  }
  // Load our Python files into the virtual FS (with cache-busting)
  const files = {
    'py/hip_inverse_dynamics.py': await (await fetch(`py/hip_inverse_dynamics.py?ver=${ASSET_VERSION}`, { cache: 'no-store' })).text(),
    'py/pages_pipeline.py': await (await fetch(`py/pages_pipeline.py?ver=${ASSET_VERSION}`, { cache: 'no-store' })).text(),
  };
  for (const [path, text] of Object.entries(files)) {
    const parts = path.split('/'); let dir = '';
    for (let i=0;i<parts.length-1;i++){ dir += (i?'/':'') + parts[i]; try{ pyodide.FS.mkdir(dir); }catch(e){} }
    pyodide.FS.writeFile(path, text);
  }
  await pyodide.runPythonAsync(`import sys; sys.path.append('py')`);
  document.getElementById('status').textContent = 'Python ready.';
  return pyodide;
}

async function runAnalysis() {
  try {
  const runBtn = document.getElementById('runBtn');
  runBtn.disabled = true; runBtn.textContent = 'Running...';
    await loadPyodideAndPackages();
    const status = document.getElementById('status');
    status.textContent = 'Reading files...';

    const fPelvis = document.getElementById('pelvis').files[0];
    const fLTh = document.getElementById('lthigh').files[0];
    const fRTh = document.getElementById('rthigh').files[0];
    const fLTb = document.getElementById('ltibia').files[0];
    const fRTb = document.getElementById('rtibia').files[0];
    if (!(fPelvis && fLTh && fRTh && fLTb && fRTb)) {
      status.textContent = 'Please select all five CSV files.'; return;
    }
    const height = parseFloat(document.getElementById('height').value || '1.70');
    const mass = parseFloat(document.getElementById('mass').value || '95.25');
    const doCal = document.getElementById('doCal').checked;
    const doOverlap = document.getElementById('doOverlap').checked;
  const showStance = document.getElementById('showStance').checked;
  const fastMode = document.getElementById('fastMode').checked;
  const gpuAccel = document.getElementById('gpuAccel')?.checked;

    // Write files to Pyodide FS
  const writeFile = async (file, path) => {
      const buf = await file.arrayBuffer();
      pyodide.FS.writeFile(path, new Uint8Array(buf));
    };
  console.time('fs:write');
  await writeFile(fPelvis, '/tmp/pelvis.csv');
    await writeFile(fLTh,    '/tmp/L_thigh.csv');
    await writeFile(fRTh,    '/tmp/R_thigh.csv');
    await writeFile(fLTb,    '/tmp/L_tibia.csv');
    await writeFile(fRTb,    '/tmp/R_tibia.csv');
  console.timeEnd('fs:write');

    status.textContent = 'Running analysis...';
    const code = `
import json
from pages_pipeline import process_files
res = process_files(
    pelvis='/tmp/pelvis.csv',
    L_thigh='/tmp/L_thigh.csv',
    R_thigh='/tmp/R_thigh.csv',
    L_tibia='/tmp/L_tibia.csv',
    R_tibia='/tmp/R_tibia.csv',
    height=${height},
    mass=${mass},
  do_cal=${doCal ? 'True' : 'False'},
  do_overlap=${doOverlap ? 'True' : 'False'},
  fast_mode=${fastMode ? 'True' : 'False'}
)
json.dumps(res)
`;
  console.time('py:process_files');
  const resultJSON = await pyodide.runPythonAsync(code);
  console.timeEnd('py:process_files');
    let res = JSON.parse(resultJSON);

    // Optional GPU acceleration: rotate FreeAcc by quaternion if provided back
    // (We keep it optional and non-blocking; falls back if WebGPU not supported.)
    if (gpuAccel) {
      try {
        await initWebGPU();
        if (res.q_flat && res.acc_flat) {
          console.time('gpu:rotate');
          const aWorld = await rotateVecsGPU(new Float32Array(res.q_flat), new Float32Array(res.acc_flat));
          console.timeEnd('gpu:rotate');
          // Currently used internally in Python; GPU path is ready for future fusion.
        }
      } catch (_) { /* ignore and continue */ }
    }
    status.textContent = 'Done.';

    // Build downloads
  const leftCsv = new Blob([res.left_csv], {type:'text/csv'});
  if (leftURL) URL.revokeObjectURL(leftURL);
  leftURL = URL.createObjectURL(leftCsv);
  document.getElementById('dlLeft').href = leftURL;
  const rightCsv = new Blob([res.right_csv], {type:'text/csv'});
  if (rightURL) URL.revokeObjectURL(rightURL);
  rightURL = URL.createObjectURL(rightCsv);
  document.getElementById('dlRight').href = rightURL;

    // Charts
    const tsCtx = document.getElementById('tsChart').getContext('2d');
    const cCtx = document.getElementById('cycleChart').getContext('2d');
  // Downsample time series for plotting if very long
  const dsStep = Math.max(1, Math.floor(res.time_s.length / 2000));
  const ds = (arr) => arr.filter((_, i) => i % dsStep === 0);
  const time_ds = ds(res.time_s);
  const left_ts_ds = ds(res.left_ts);
  const right_ts_ds = ds(res.right_ts);
    const commonOptions = {
      responsive: true, animation: false,
      scales: { x: { title: { display: true } }, y: { title: { display: true, text: 'Hip moment My (N·m)' } } },
      plugins: { legend: { labels: { color: '#e6eaff' } } }
    };
    // Time-series overlay
    const tsData = {
      labels: time_ds,
      datasets: [
        { label: 'Left hip', data: left_ts_ds, borderColor: '#6ea8fe', fill: false, pointRadius: 0 },
        { label: 'Right hip', data: right_ts_ds, borderColor: '#ff8fa3', fill: false, pointRadius: 0 }
      ]
    };
    if (tsChart) tsChart.destroy();
    tsChart = new Chart(tsCtx, {
      type: 'line', data: tsData,
      options: { ...commonOptions, scales: { ...commonOptions.scales, x: { title: { display: true, text: 'Time (s)' } } } }
    });

    // Cycle-normalized overlay with SD
    const pct = res.cycle_pct;
    const Lmean = res.left_mean, Lsd = res.left_sd;
    const Rmean = res.right_mean, Rsd = res.right_sd;
    const cycleData = {
      labels: pct,
      datasets: [
        { label: 'Left mean', data: Lmean, borderColor: '#6ea8fe', fill: false, pointRadius: 0 },
        { label: 'Left +SD', data: Lmean.map((y,i)=>y+Lsd[i]), borderColor: 'rgba(110,168,254,0.0)', backgroundColor: 'rgba(110,168,254,0.2)', fill: '+1', pointRadius: 0 },
        { label: 'Left -SD', data: Lmean.map((y,i)=>y-Lsd[i]), borderColor: 'rgba(110,168,254,0.0)', backgroundColor: 'rgba(110,168,254,0.2)', fill: false, pointRadius: 0 },
        { label: 'Right mean', data: Rmean, borderColor: '#ff8fa3', fill: false, pointRadius: 0 },
        { label: 'Right +SD', data: Rmean.map((y,i)=>y+Rsd[i]), borderColor: 'rgba(255,143,163,0.0)', backgroundColor: 'rgba(255,143,163,0.2)', fill: '+1', pointRadius: 0 },
        { label: 'Right -SD', data: Rmean.map((y,i)=>y-Rsd[i]), borderColor: 'rgba(255,143,163,0.0)', backgroundColor: 'rgba(255,143,163,0.2)', fill: false, pointRadius: 0 }
      ]
    };
    if (cycleChart) cycleChart.destroy();
    cycleChart = new Chart(cCtx, {
      type: 'line', data: cycleData,
      options: { ...commonOptions, scales: { ...commonOptions.scales, x: { title: { display: true, text: 'Gait cycle (%)' } } } }
    });

  } catch (err) {
    console.error(err);
    document.getElementById('status').textContent = 'Error: ' + err.message;
  } finally {
    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = false; runBtn.textContent = 'Run analysis';
  }
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('runBtn').addEventListener('click', runAnalysis);
  // Warm up Pyodide in the background so it's ready by the time the user selects files
  try { loadPyodideAndPackages(); } catch(e) { /* ignore */ }
  // Register service worker for caching Pyodide and app shell
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(()=>{});
  }
});
