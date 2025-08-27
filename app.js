let pyodide = null;
let tsChart = null, cycleChart = null;
let leftURL = null, rightURL = null;

async function loadPyodideAndPackages() {
  if (pyodide) return pyodide;
  document.getElementById('status').textContent = 'Loading Python (Pyodide)...';
  pyodide = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/' });
  // Load required Python packages from the Pyodide distribution
  document.getElementById('status').textContent = 'Loading Python packages (numpy, pandas)...';
  try {
    await pyodide.loadPackage([ 'numpy', 'pandas' ]);
  } catch (e) {
    console.error('Failed to load Pyodide packages', e);
    document.getElementById('status').textContent = 'Error loading Python packages: ' + e.message;
    throw e;
  }
  // Load our Python files into the virtual FS
  const files = {
    'py/hip_inverse_dynamics.py': await (await fetch('py/hip_inverse_dynamics.py')).text(),
    'py/pages_pipeline.py': await (await fetch('py/pages_pipeline.py')).text(),
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

    // Write files to Pyodide FS
    const writeFile = async (file, path) => {
      const buf = await file.arrayBuffer();
      pyodide.FS.writeFile(path, new Uint8Array(buf));
    };
    await writeFile(fPelvis, '/tmp/pelvis.csv');
    await writeFile(fLTh,    '/tmp/L_thigh.csv');
    await writeFile(fRTh,    '/tmp/R_thigh.csv');
    await writeFile(fLTb,    '/tmp/L_tibia.csv');
    await writeFile(fRTb,    '/tmp/R_tibia.csv');

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
  do_overlap=${doOverlap ? 'True' : 'False'}
)
json.dumps(res)
`;
    const resultJSON = await pyodide.runPythonAsync(code);
    const res = JSON.parse(resultJSON);
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
    const commonOptions = {
      responsive: true, animation: false,
      scales: { x: { title: { display: true } }, y: { title: { display: true, text: 'Hip moment My (N·m)' } } },
      plugins: { legend: { labels: { color: '#e6eaff' } } }
    };
    // Time-series overlay
    const tsData = {
      labels: res.time_s,
      datasets: [
        { label: 'Left hip', data: res.left_ts, borderColor: '#6ea8fe', fill: false, pointRadius: 0 },
        { label: 'Right hip', data: res.right_ts, borderColor: '#ff8fa3', fill: false, pointRadius: 0 }
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
});
