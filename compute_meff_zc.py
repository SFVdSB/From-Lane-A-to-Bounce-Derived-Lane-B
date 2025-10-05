#!/usr/bin/env python3
"""
compute_meff_zc.py
Compute the light Hessian eigenvalue λ_min along the bounce track, convert to a physical
m_eff, and predict the onset redshift z_c from the crossing m_eff = 3 H(z).
Also identify:
  - "half-max (falling) point" based on ΔV along the track (direction toward the false vacuum),
  - "global min-λ point",
  - "best-match point" with λ_min closest to the λ implied by a target z_c (optional).
"""

import json, math, numpy as np, pandas as pd, importlib.util, argparse
from pathlib import Path

def load_gold(path):
    spec = importlib.util.spec_from_file_location("golden", str(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def V_fun(gold, p):
    def V(Phi, phi): return float(gold.potential_rescaled(float(Phi), float(phi), p))
    return V

def smooth(y, win):
    if win<=1: return y
    win = int(win) + (int(win)%2==0)
    pad = win//2
    yp = np.pad(y, (pad,pad), mode="edge"); ker = np.ones(win)/win
    return np.convolve(yp, ker, mode="valid")

def hess_fd(V, P0, p0, hP=None, hp=None):
    hP = hP or (1e-4 * max(1.0, abs(P0))); hp = hp or (1e-4 * max(1.0, abs(p0)))
    V00 = V(P0,p0); VPP = V(P0+hP,p0); VMM = V(P0-hP,p0); Vpp = V(P0,p0+hp); Vmm = V(P0,p0-hp)
    V_PM = V(P0+hP,p0+hp); V_Pm = V(P0+hP,p0-hp); V_mP = V(P0-hP,p0+hp); V_mm = V(P0-hP,p0-hp)
    V_PP = (VPP - 2*V00 + VMM)/(hP*hP); V_pp = (Vpp - 2*V00 + Vmm)/(hp*hp)
    V_Pp = (V_PM - V_Pm - V_mP + V_mm)/(4*hP*hp)
    lam = np.linalg.eigvalsh(np.array([[V_PP, V_Pp],[V_Pp, V_pp]], float))
    return float(lam[0]), float(lam[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default="background_profile.csv")
    ap.add_argument("--gold", default="goldenRunDetails_v4f_more3.py")
    ap.add_argument("--pt_json", default="p_t_template.json")
    ap.add_argument("--H0", type=float, default=72.0)
    ap.add_argument("--Om", type=float, default=0.322)
    ap.add_argument("--Neff", type=float, default=3.046)
    ap.add_argument("--v_phys_planck", type=float, default=4.2e-5)
    ap.add_argument("--target_zc", type=float, default=15500.0)
    ap.add_argument("--out", default="meff_onset_summary.json")
    args = ap.parse_args()

    bg = pd.read_csv(args.bg)
    gold = load_gold(args.gold)
    with open(args.pt_json,"r") as f: p = json.load(f)
    V = V_fun(gold, p)

    r = bg["r"].to_numpy(float); P = bg["Phi"].to_numpy(float); s = bg["phi"].to_numpy(float)

    # ΔV and half-max (falling side toward false vacuum)
    V_arr = np.array([V(P[i], s[i]) for i in range(len(r))], float)
    V_false = float(np.mean(V_arr[-max(5, len(V_arr)//50):]))
    dV = np.maximum(V_arr - V_false, 0.0)
    dV_s = smooth(dV, max(7, len(dV)//150))
    i_pk = int(np.argmax(dV_s)); half = 0.5*float(dV_s[i_pk])
    # decide which tail is the false vacuum (smaller ΔV)
    tail_left = float(np.mean(dV_s[:max(3, len(dV_s)//20)]))
    tail_right = float(np.mean(dV_s[-max(3, len(dV_s)//20):]))
    if tail_right < tail_left:
        i_half = next((j+1 for j in range(i_pk, len(dV_s)-1) if dV_s[j] >= half and dV_s[j+1] < half), min(len(dV_s)-2, i_pk+max(1, len(dV_s)//20)))
    else:
        i_half = next((j-1 for j in range(i_pk, 0, -1) if dV_s[j] >= half and dV_s[j-1] < half), max(1, i_pk-max(1, len(dV_s)//20)))
    i_half = int(np.clip(i_half, 0, len(P)-1))

    # Scan downsampled for min λ
    step = max(1, len(P)//800)
    mins = []
    for i in range(0, len(P), step):
        lam_min, lam_max = hess_fd(V, P[i], s[i])
        if lam_min>0: mins.append((lam_min, i))
    lam_coarse, i_coarse = sorted(mins, key=lambda t: t[0])[0]

    # refine near min
    lam_min = lam_coarse; i_min = i_coarse
    for i in range(max(0, i_coarse-1000), min(len(P), i_coarse+1000)):
        lam, _ = hess_fd(V, P[i], s[i])
        if lam>0 and lam<lam_min: lam_min, i_min = lam, i

    # Physics scales
    Lambda_star_eV = 0.72
    Mpl_eV = 2.435e27
    v_phys_eV = args.v_phys_planck*Mpl_eV
    hbar = 6.582119569e-16
    h = args.H0/100.0
    omega_gamma = 2.469e-5
    Or = (omega_gamma*(1.0 + 0.2271*args.Neff))/(h*h)
    Ode = 1.0 - args.Om - Or
    H0_s = 100.0*h * 1000.0 / 3.0856775814913673e22
    H0_eV = H0_s * hbar
    def H_eV(z): return H0_eV*math.sqrt(Or*(1+z)**4 + args.Om*(1+z)**3 + Ode)

    def zc_from_lam(lam):
        me = (Lambda_star_eV**2 / v_phys_eV)*math.sqrt(max(lam,0.0))
        # solve m_eff = 3 H(z)
        z_lo, z_hi = 100.0, 2.0e6
        for _ in range(100):
            z_mid = (z_lo*z_hi)**0.5
            if 3.0*H_eV(z_mid) < me: z_lo = z_mid
            else: z_hi = z_mid
        return 0.5*(z_lo+z_hi), me

    zc_half, me_half = zc_from_lam(hess_fd(V, P[i_half], s[i_half])[0])
    zc_min, me_min = zc_from_lam(lam_min)

    # Best-match to a target z_c (optional)
    lam_needed = (3*H_eV(args.target_zc) * v_phys_eV / (Lambda_star_eV**2))**2
    # local search near min to find lam closest to lam_needed
    best = (1e9, None, None)
    i0 = i_min
    for i in range(max(0, i0-2000), min(len(P), i0+2000)):
        lam, _ = hess_fd(V, P[i], s[i])
        if lam<=0: continue
        diff = abs(lam - lam_needed)
        if diff < best[0]: best = (diff, i, lam)
    i_star = best[1]; lam_star = best[2]
    zc_star, me_star = zc_from_lam(lam_star)

    out = {
        "indices": {"i_peak": int(i_pk), "i_half": int(i_half), "i_minLam": int(i_min), "i_star": int(i_star)},
        "r_values": {"r_half": float(r[i_half]), "r_minLam": float(r[i_min]), "r_star": float(r[i_star])},
        "lambda": {"half": float(hess_fd(V, P[i_half], s[i_half])[0]), "min": float(lam_min), "star": float(lam_star), "lam_needed_for_target": float(lam_needed)},
        "m_eff_eV": {"half": me_half, "min": me_min, "star": me_star},
        "z_c": {"half": zc_half, "min": zc_min, "star": zc_star, "target": float(args.target_zc)},
        "scales": {"Lambda_star_eV": Lambda_star_eV, "v_phys_over_Mpl": args.v_phys_planck},
        "cosmo": {"H0": args.H0, "Om": args.Om, "Neff": args.Neff, "Or": Or}
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__=="__main__":
    main()
