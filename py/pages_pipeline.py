# pages_pipeline.py
# Client-side pipeline for Pyodide (browser). Uses only numpy/pandas + hip_inverse_dynamics.
# Steps: load CSVs -> optional overlap cleaning -> optional standing calibration -> hip ID -> return JSON-friendly dict.
import numpy as np, pandas as pd, io, json
from itertools import islice
from hip_inverse_dynamics import (
    SegmentKinematics, make_inertial_props, inverse_dynamics_lowerlimb_3D,
    com_from_joints_linear, GRAVITY, deleva_lower_limb_fractions
)

def _q_to_R(qw,qx,qy,qz):
    nrm = (qw*qw+qx*qx+qy*qy+qz*qz)**0.5 or 1.0
    w,x,y,z = qw/nrm, qx/nrm, qy/nrm, qz/nrm
    return np.array([[1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]])

def _read_xsens(path):
    # Detect header row where IMU fields appear and infer delimiter robustly
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = list(islice(f, 50))
    if not lines:
        raise ValueError(f"Empty or unreadable CSV: {path}")

    header_idx = None
    for i, line in enumerate(lines):
        l = line.lower()
        if ('quat' in l) or ('freeacc' in l) or ('orientation' in l):
            header_idx = i
            break
    if header_idx is None:
        # Fallback to first non-empty line if specific header not found
        header_idx = next((i for i, ln in enumerate(lines) if ln.strip()), 0)

    # Delimiter detection
    first = lines[0].strip().lower()
    if first.startswith('sep='):
        sep = first.split('=', 1)[1].strip() or ','
    else:
        hdr_line = lines[header_idx]
        sep = ',' if hdr_line.count(',') >= hdr_line.count(';') else ';'

    df = pd.read_csv(path, sep=sep, header=header_idx, engine='python')
    # Sanitize column names to match patterns robustly
    orig_cols = list(df.columns)
    def sanitize(name: str) -> str:
        s = str(name).strip().lower()
        for ch in [' ', '\\t', '(', ')', '[', ']', '{', '}', '/', '\\', '-', ':']:
            s = s.replace(ch, '_')
        while '__' in s:
            s = s.replace('__', '_')
        return s.strip('_')
    san_cols = [sanitize(c) for c in orig_cols]
    # map sanitized -> original for retrieval
    san_to_orig = {s:o for s,o in zip(san_cols, orig_cols)}
    def find_col(candidates, required=True):
        for cand in candidates:
            if cand in san_to_orig:
                return san_to_orig[cand]
            # allow prefix match (e.g., freeacc_x_ms2)
            for s in san_cols:
                if s.startswith(cand):
                    return san_to_orig[s]
        if required:
            raise ValueError(f"Missing required column. Tried any of: {candidates} in {path}")
        return None
    # Resolve quaternion columns
    qw_col = find_col(['quat_w','quaternion_w','orientation_w','ori_w','qw'])
    qx_col = find_col(['quat_x','quaternion_x','orientation_x','ori_x','qx'])
    qy_col = find_col(['quat_y','quaternion_y','orientation_y','ori_y','qy'])
    qz_col = find_col(['quat_z','quaternion_z','orientation_z','ori_z','qz'])
    # Resolve FreeAcc columns (world frame)
    fax_col = find_col(['freeacc_x','free_acc_x','acc_free_x','free_acceleration_x'])
    fay_col = find_col(['freeacc_y','free_acc_y','acc_free_y','free_acceleration_y'])
    faz_col = find_col(['freeacc_z','free_acc_z','acc_free_z','free_acceleration_z'])
    # Optional time column
    stf_col = find_col(['sampletimefine','sample_time_fine','sample_time_fine_ticks','time_ms','timestamp'], required=False)

    df.columns = orig_cols  # keep original for pandas ops
    qw = df[qw_col].to_numpy(float); qx=df[qx_col].to_numpy(float)
    qy = df[qy_col].to_numpy(float); qz=df[qz_col].to_numpy(float)
    R = np.array([_q_to_R(qw[i],qx[i],qy[i],qz[i]) for i in range(len(df))])
    if stf_col is not None:
        # Heuristic: SampleTimeFine is in 0.1 ms ticks (10 kHz). If value seems very large, divide accordingly
        t_raw = df[stf_col].to_numpy(float)
        t = t_raw / (1e4 if t_raw.max() > 1e3 else 1.0)
        t = t - t[0]
        inc = np.diff(t, prepend=t[0]-1e-6) > 0
        if not inc.all():
            # Drop duplicates/non-increasing
            df = df[inc]; t = t[inc]; R = R[inc]
    else:
        t = np.arange(len(df))/60.0
    # world-frame "FreeAcc"
    ax=df[fax_col].to_numpy(float); ay=df[fay_col].to_numpy(float); az=df[faz_col].to_numpy(float)
    Ab = np.column_stack([ax,ay,az])
    Aw = np.einsum('tij,tj->ti', R, Ab)
    # angular vel from Rdot
    if len(t)>1:
        dt = np.gradient(t)
    else:
        dt = np.array([1/60.0])
    Rdot = np.gradient(R, axis=0) / dt[:,None,None]
    omega = np.zeros((len(t),3))
    for i in range(len(t)):
        Wm = Rdot[i] @ R[i].T
        omega[i] = np.array([Wm[2,1]-Wm[1,2], Wm[0,2]-Wm[2,0], Wm[1,0]-Wm[0,1]])*0.5
    return dict(t=t, R=R, omega=omega, acc=Aw)

def _overlap_window(ds):
    starts = [d["t"][0] for d in ds]
    ends   = [d["t"][-1] for d in ds]
    return max(starts), min(ends)

def _trim(d, t0, t1):
    m = (d["t"]>=t0) & (d["t"]<=t1)
    out = {k:(v[m].copy() if hasattr(v,'__len__') else v) for k,v in d.items()}
    return out

def _avg_rot(Rs):
    M = Rs.mean(axis=0)
    U,S,Vt = np.linalg.svd(M)
    Ravg = U @ Vt
    if np.linalg.det(Ravg) < 0:
        U[:, -1] *= -1
        Ravg = U @ Vt
    return Ravg

def _static_mask(omega, dt, window_s=3.0, thresh=0.2):
    n = len(omega)
    w = max(1,int(window_s/dt))
    beg = np.arange(0,min(n,w)); end=np.arange(max(0,n-w),n)
    wn = np.linalg.norm(omega,axis=1)
    m = np.zeros(n,dtype=bool); m[beg]=wn[beg]<thresh; m[end]=wn[end]<thresh
    return m

def _calibrate(d):
    dt = np.median(np.diff(d["t"])) if len(d["t"])>1 else 1/60.0
    m = _static_mask(d["omega"], dt)
    Rs = d["R"][m] if m.any() else d["R"][:max(20,int(0.5/dt))]
    Rstat = _avg_rot(Rs)
    C = Rstat.T
    Rcal = np.einsum('tij,jk->tik', d["R"], C)
    # recompute omega
    if len(d["t"])>1:
        dt_arr = np.gradient(d["t"])
    else:
        dt_arr = np.array([1/60.0])
    Rdot = np.gradient(Rcal, axis=0) / dt_arr[:,None,None]
    omega = np.zeros_like(d["omega"])
    for i in range(len(omega)):
        Wm = Rdot[i] @ Rcal[i].T
        omega[i] = np.array([Wm[2,1]-Wm[1,2], Wm[0,2]-Wm[2,0], Wm[1,0]-Wm[0,1]])*0.5
    # rotate FreeAcc using calibrated R (first infer body approx then reapply)
    Ab = np.einsum('tij,tj->ti', np.transpose(d["R"], (0,2,1)), d["acc"])
    Aw = np.einsum('tij,tj->ti', Rcal, Ab)
    return {"t": d["t"], "R": Rcal, "omega": omega, "acc": Aw}

def _build_chain(R_thigh, R_tibia, pelvis_h, L_thigh, L_shank, L_foot):
    T = R_thigh.shape[0]
    hip = np.tile(np.array([0,0,pelvis_h]), (T,1))
    knee = np.zeros((T,3)); ankle = np.zeros((T,3)); toe = np.zeros((T,3))
    thigh_dir0 = np.array([0,0,-1.0]); shank_dir0=np.array([0,0,-1.0]); foot_dir0=np.array([1.0,0,0])
    for i in range(T):
        knee[i]  = hip[i]   + R_thigh[i] @ (thigh_dir0*L_thigh)
        ankle[i] = knee[i]  + R_tibia[i] @ (shank_dir0*L_shank)
        toe[i]   = ankle[i] + R_tibia[i] @ (foot_dir0 *L_foot)
    return hip, knee, ankle, toe, R_tibia.copy()

def _detect_stance(accW, omegaW, acc_thresh=2.0, omega_thresh=1.5):
    a = np.linalg.norm(accW,axis=1); w=np.linalg.norm(omegaW,axis=1)
    s=((a<acc_thresh)&(w<omega_thresh)).astype(float)
    ker=np.ones(5)/5.0
    return np.convolve(s,ker,mode='same')

def _rocker(s, N):
    # Build monotonic 0->1 over each stance region; if none, simple linspace to avoid crashes
    u = np.zeros(N); on=False; t0=0
    for i,val in enumerate(s>0.5):
        if val and not on: on=True; t0=i
        if (not val) and on:
            dur = i-t0
            if dur>1:
                u[t0:i] = np.linspace(0,1,dur)
            on=False
    if on:
        dur = N - t0
        if dur>1:
            u[t0:] = np.linspace(0,1,dur)
    if not (u>0).any():
        u = np.linspace(0,1,N)  # fallback
    return u

def _cycle_norm(t, y, stance):
    s = (stance>0.5).astype(int)
    ds = np.diff(s, prepend=s[0])
    hs = np.where(ds==1)[0]
    if len(hs)<2: return None,None,None
    curves=[]
    for i in range(len(hs)-1):
        a,b = hs[i], hs[i+1]
        if b<=a+5: continue
        tt = t[a:b]-t[a]
        yy = y[a:b]
        pct = np.linspace(0,1,101)
        yn = np.interp(pct, tt/tt[-1], yy)
        curves.append(yn)
    if len(curves)==0: return None,None,None
    M = np.vstack(curves)
    return (np.linspace(0,100,101), M.mean(axis=0), M.std(axis=0))

def process_files(pelvis, L_thigh, R_thigh, L_tibia, R_tibia, height, mass, do_cal=True, do_overlap=True):
    # 1) Load
    P = _read_xsens(pelvis)
    LTh = _read_xsens(L_thigh)
    RTh = _read_xsens(R_thigh)
    LTi = _read_xsens(L_tibia)
    RTi = _read_xsens(R_tibia)

    # 2) Overlap cleaning
    if do_overlap:
        t0,t1 = _overlap_window([P,LTh,RTh,LTi,RTi])
        P = _trim(P,t0,t1); LTh=_trim(LTh,t0,t1); RTh=_trim(RTh,t0,t1); LTi=_trim(LTi,t0,t1); RTi=_trim(RTi,t0,t1)
        # equalize counts
        N = min(len(P["t"]),len(LTh["t"]),len(RTh["t"]),len(LTi["t"]),len(RTi["t"]))
        for d in (P,LTh,RTh,LTi,RTi):
            for k in ("t","R","omega","acc"):
                d[k]=d[k][:N]

    # 3) Standing calibration
    if do_cal:
        P = _calibrate(P)
        LTh = _calibrate(LTh); RTh = _calibrate(RTh)
        LTi = _calibrate(LTi); RTi = _calibrate(RTi)

    # 4) Kinematics model
    scale = height/1.75
    L_th = 0.45*scale; L_sh=0.43*scale; L_fo=0.25*scale; pelvis_h=0.95*scale; heel_to_ankle = 0.07*scale
    fr = deleva_lower_limb_fractions()

    def run_side(Th, Ti):
        Tn = len(Th["t"]); dt = np.median(np.diff(Th["t"])) if Tn>1 else 1/60.0
        hip,knee,ankle,toe,Rfoot = _build_chain(Th["R"], Ti["R"], pelvis_h, L_th, L_sh, L_fo)
        rcom_th = com_from_joints_linear(hip, knee,  fr["thigh"]["com"])
        rcom_sh = com_from_joints_linear(knee, ankle, fr["shank"]["com"])
        rcom_fo = com_from_joints_linear(ankle,toe,   fr["foot"]["com"])
        a_th, a_sh, a_fo = Th["acc"], Ti["acc"], Ti["acc"]
        stance = _detect_stance(Ti["acc"], Ti["omega"])
        u = _rocker(stance, Tn)
        # GRF
        Ftot = np.c_[mass*P["acc"][:,0], mass*P["acc"][:,1], mass*(P["acc"][:,2]+GRAVITY[2])]
        Fw = stance[:,None]*Ftot
        # CoP
        x_heel = -heel_to_ankle; x_toe = L_fo-heel_to_ankle
        rcop = np.zeros((Tn,3))
        for i in range(Tn):
            copF = np.array([(1-u[i])*x_heel + u[i]*x_toe, 0, 0])
            rcop[i] = Rfoot[i] @ copF + ankle[i]
        # props
        prop_fo  = make_inertial_props(height,mass,L_fo,"foot")
        prop_sh  = make_inertial_props(height,mass,L_sh,"shank")
        prop_th  = make_inertial_props(height,mass,L_th,"thigh")
        # dynamics
        hipM_world = np.zeros((Tn,3))
        alpha_th = np.gradient(Th["omega"], axis=0) / (np.gradient(Th["t"])[:,None] if Tn>1 else 1/60.0)
        alpha_ti = np.gradient(Ti["omega"], axis=0) / (np.gradient(Ti["t"])[:,None] if Tn>1 else 1/60.0)
        for i in range(Tn):
            kinF = SegmentKinematics(R_WB=Rfoot[i], omega_W=Ti["omega"][i], alpha_W=alpha_ti[i],
                                     r_COM_W=rcom_fo[i], a_COM_W=a_fo[i],
                                     r_prox_W=ankle[i], r_dist_W=toe[i])
            kinS = SegmentKinematics(R_WB=Ti["R"][i], omega_W=Ti["omega"][i], alpha_W=alpha_ti[i],
                                     r_COM_W=rcom_sh[i], a_COM_W=a_sh[i],
                                     r_prox_W=knee[i], r_dist_W=ankle[i])
            kinT = SegmentKinematics(R_WB=Th["R"][i], omega_W=Th["omega"][i], alpha_W=alpha_th[i],
                                     r_COM_W=rcom_th[i], a_COM_W=a_th[i],
                                     r_prox_W=hip[i], r_dist_W=knee[i])
            loads = inverse_dynamics_lowerlimb_3D(kinF, kinS, kinT, prop_fo, prop_sh, prop_th,
                                                  F_GRF_W=Fw[i], r_CoP_W=rcop[i], M_free_W=None)
            hipM_world[i] = loads["hip"].M
        M_thigh = np.einsum('tji,ti->tj', Th["R"], hipM_world)
        return Th["t"], M_thigh[:,1], stance

    tL, Mleft, stanceL = run_side(LTh, LTi)
    tR, Mright, stanceR = run_side(RTh, RTi)

    # Cycle stats
    pctL, meanL, sdL = _cycle_norm(tL, Mleft, stanceL)
    pctR, meanR, sdR = _cycle_norm(tR, Mright, stanceR)
    if pctL is None or pctR is None:
        pct = list(np.linspace(0,100,101))
        meanL_list = sdL_list = meanR_list = sdR_list = [float('nan')]*101
    else:
        pct = list(pctL)  # assume both are 0..100 same
        meanL_list = list(meanL)
        sdL_list = list(sdL)
        meanR_list = list(meanR)
        sdR_list = list(sdR)

    # CSVs
    left_df = pd.DataFrame({'time_s': tL, 'hip_My_Nm': Mleft})
    right_df= pd.DataFrame({'time_s': tR, 'hip_My_Nm': Mright})
    left_csv = left_df.to_csv(index=False)
    right_csv= right_df.to_csv(index=False)

    return dict(
        time_s=list(tL),
        left_ts=list(Mleft),
        right_ts=list(Mright),
        cycle_pct=pct,
    left_mean=meanL_list,
    left_sd=sdL_list,
    right_mean=meanR_list,
    right_sd=sdR_list,
        left_csv=left_csv,
        right_csv=right_csv,
    )
