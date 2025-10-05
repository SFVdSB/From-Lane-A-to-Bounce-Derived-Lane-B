
#!/usr/bin/env python3
# laneB_with_SFVclock.py
# Lane B (two-field FRW) + optional SFV clock rescaling near equality.
# Adds:
#   --sfv-clock {none,eq,gauss}
#   --eps <float>                : amplitude (for eq) or A (for gauss)
#   --gauss-zc <float>           : center (default 3000)
#   --gauss-sig <float>          : sigma in ln(1+z) (default 0.30)
#
# The clock multiplies H/H0 by alpha(z), i.e. E(z) -> alpha(z)*E(z).
# This modifies r_d while leaving low-z distances ~unchanged for the eq-shape (fr*fm).
#
# DESI BAO mapping uses Hz*rd/C_KMS (dimensionless).

import argparse, os, json, numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict
from scipy import integrate

C_KMS = 299792.458  # km/s

# ---------------- Cosmology helpers ----------------
@dataclass
class CosmoParams:
    H0: float = 72.5
    Om: float = 0.30
    Obh2: float = 0.0224
    Neff: float = 3.046
    def Orad(self) -> float:
        h = self.H0/100.0
        Og = 2.469e-5/(h*h)
        return Og * (1.0 + 0.2271*self.Neff)

def E_LCDM(z, Om, Orad, Ode):
    return np.sqrt(Orad*(1+z)**4 + Om*(1+z)**3 + Ode)

# ---------------- SFV clock alpha(z) ----------------
def alpha_eqshape(z, Om, Orad, Ode, eps):
    # fr = Orad*(1+z)^4 / E^2, fm = Om*(1+z)^3 / E^2
    E2 = Orad*(1+z)**4 + Om*(1+z)**3 + Ode
    fr = Orad*(1+z)**4 / E2
    fm = Om*(1+z)**3 / E2
    return 1.0 + eps * (fr*fm)

def alpha_gauss(z, A=0.08, zc=3000.0, sigma_ln1pz=0.30):
    x = (np.log(1.0+z) - np.log(1.0+zc)) / sigma_ln1pz
    return 1.0 + A * np.exp(-0.5 * x * x)

def make_E_with_clock(Om, Orad, Ode, base_E:Callable[[float],float], mode:str, eps:float, zc:float, sig:float):
    if mode == "none":
        return base_E
    def Ez(z):
        z = float(z)
        if mode == "eq":
            a = alpha_eqshape(z, Om, Orad, Ode, eps)
        elif mode == "gauss":
            a = alpha_gauss(z, A=eps, zc=zc, sigma_ln1pz=sig)
        else:
            a = 1.0
        return float(base_E(z) * a)
    return Ez

# ---------------- Distances and r_d ----------------
def distance_comoving_Mpc(z, H0, Ez_func):
    integ, _ = integrate.quad(lambda zz: 1.0/float(Ez_func(zz)), 0.0, float(z), epsabs=1e-7, epsrel=1e-6, limit=500)
    return (C_KMS/H0)*integ

def z_drag_EH(Obh2, Omh2):
    b1 = 0.313 * (Omh2)**(-0.419) * (1 + 0.607*(Omh2)**0.674)
    b2 = 0.238 * (Omh2)**0.223
    return 1291 * (Omh2)**0.251 / (1 + 0.659*(Omh2)**0.828) * (1 + b1*(Obh2)**b2)

def r_d_Mpc_from_E(H0, Om, Orad, Obh2, Ez_callable):
    h = H0/100.0
    zd = z_drag_EH(Obh2, Om*h*h)
    Og = 2.469e-5/(h*h)
    Ob = Obh2/(h*h)
    def integrand(z):
        Ez = max(float(Ez_callable(z)), 1e-10)
        R = 3.0*Ob/(4.0*Og) * 1.0/(1.0+z)
        cs_over_c = 1.0/np.sqrt(3.0*(1.0+R))
        return cs_over_c / Ez
    I, _ = integrate.quad(integrand, zd, np.inf, epsabs=1e-7, epsrel=1e-6, limit=500)
    return (C_KMS/H0)*I

# ---------------- DESI BAO ----------------
def bao_predictions(z, H0, Ez_callable):
    z = np.atleast_1d(z).astype(float)
    Ez = np.array([float(Ez_callable(zi)) for zi in z])
    Hz = H0 * Ez
    DC = np.array([distance_comoving_Mpc(zi, H0, Ez_callable) for zi in z])
    DM = DC
    DH = C_KMS / np.clip(Hz, 1e-12, 1e12)
    DV = ((DM**2) * (C_KMS*z/np.clip(Hz,1e-12,1e12)))**(1.0/3.0)
    return {"DV": DV, "DM": DM, "DH": DH, "Hz": Hz}

def parse_bao_mean_cov(mean_path, cov_path):
    if not (os.path.exists(mean_path) and os.path.exists(cov_path)):
        raise FileNotFoundError("DESI mean/cov files not found.")
    entries = []
    with open(mean_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"): continue
            parts = s.split()
            z = float(parts[0]); val = float(parts[1]); lab = parts[2].strip().lower()
            entries.append({"z": z, "val": val, "label": lab})
    Cov = np.loadtxt(cov_path)
    return entries, Cov

def build_bao_model_vector(entries, H0, Ez_callable, rd):
    obs = bao_predictions(np.array([e["z"] for e in entries]), H0, Ez_callable)
    label_map = {
        "dv_over_rs": ("DV", lambda arr: arr/rd),
        "dm_over_rs": ("DM", lambda arr: arr/rd),
        "dh_over_rs": ("DH", lambda arr: arr/rd),
        "hz_times_rs": ("Hz", lambda arr: arr*rd/C_KMS),
        "hz_times_rd": ("Hz", lambda arr: arr*rd/C_KMS),
        "dv_over_rd": ("DV", lambda arr: arr/rd),
        "dm_over_rd": ("DM", lambda arr: arr/rd),
        "dh_over_rd": ("DH", lambda arr: arr/rd),
    }
    m = []
    for i,e in enumerate(entries):
        key, fn = label_map[e["label"]]
        m.append(fn(obs[key])[i])
    return np.array(m, float)

def chi2_full(y, m, Cov):
    C = 0.5*(Cov + Cov.T)
    iC = np.linalg.pinv(C, rcond=1e-12)
    r = y - m
    return float(r @ iC @ r), iC

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H0", type=float, default=72.5)
    ap.add_argument("--Om", type=float, default=0.30)
    ap.add_argument("--Obh2", type=float, default=0.0224)
    ap.add_argument("--Neff", type=float, default=3.046)
    ap.add_argument("--sfv-clock", choices=["none","eq","gauss"], default="none")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--gauss-zc", type=float, default=3000.0)
    ap.add_argument("--gauss-sig", type=float, default=0.30)

    ap.add_argument("--bao_mean", type=str, default="desi_2024_gaussian_bao_ALL_GCcomb_mean.txt")
    ap.add_argument("--bao_cov", type=str,  default="desi_2024_gaussian_bao_ALL_GCcomb_cov.txt")
    ap.add_argument("--sne", type=str, default="sne_pantheonplus_relative_STAT.csv")
    ap.add_argument("--use-sne", action="store_true")
    ap.add_argument("--fit-rd", action="store_true")
    ap.add_argument("--plot", action="store_true")

    args = ap.parse_args()
    Orad = CosmoParams(H0=args.H0, Om=args.Om, Obh2=args.Obh2, Neff=args.Neff).Orad()
    Ode = 1.0 - args.Om - Orad
    if Ode <= 0: raise SystemExit(f"Ode<=0 (got {Ode:.4f})")

    base_E = lambda z: E_LCDM(z, args.Om, Orad, Ode)
    Ez = make_E_with_clock(args.Om, Orad, Ode, base_E, args.sfv_clock, args.eps, args.gauss_zc, args.gauss_sig)

    # r_d with (possibly) modified early-time H(z)
    rd = r_d_Mpc_from_E(args.H0, args.Om, Orad, args.Obh2, Ez)

    # BAO
    entries, Cov = parse_bao_mean_cov(args.bao_mean, args.bao_cov)
    y = np.array([e["val"] for e in entries], float)
    m = build_bao_model_vector(entries, args.H0, Ez, rd)
    chi2, iC = chi2_full(y, m, Cov)

    fit_dict = "skipped"
    if args.fit_rd:
        denom = float(m @ iC @ m)
        alpha = float(y @ iC @ m)/denom if denom!=0.0 else 1.0
        chi2_best = float((y - alpha*m) @ iC @ (y - alpha*m))
        rd_best = rd / (alpha if alpha!=0.0 else 1.0)
        fit_dict = {"alpha": alpha, "r_d_best_Mpc": rd_best, "chi2_best": chi2_best}

    # Optional SNe (stat-only)
    def sne_stats(csv_path, H0, Ez_callable):
        import pandas as pd
        if not os.path.exists(csv_path): return None
        df = pd.read_csv(csv_path)
        z = df["z"].to_numpy(float)
        mu = df["mu"].to_numpy(float)
        sig = df["sigma_mu"].to_numpy(float)
        DL = (1.0+z) * np.array([distance_comoving_Mpc(zi, H0, Ez_callable) for zi in z])
        mu0 = 5.0*np.log10(np.maximum(DL, 1e-30)) + 25.0
        w = 1.0/np.maximum(sig*sig, 1e-12)
        M = float(np.sum(w*(mu - mu0))/np.sum(w))
        chi2 = float(np.sum(w*(mu - (mu0 + M))**2))
        return {"M_shift_best": M, "chi2": chi2, "N": int(len(z))}
    sne_dict = sne_stats(args.sne, args.H0, Ez) if args.use_sne else "skipped"

    out = {
        "LaneB_SFVclock": {
            "sfv_clock": args.sfv_clock,
            "eps": args.eps,
            "gauss_zc": args.gauss_zc,
            "gauss_sig": args.gauss_sig,
            "H0": args.H0,
            "Om": args.Om,
            "Orad": Orad,
            "Ode": Ode,
            "r_d_Mpc": rd
        },
        "BAO_fullcov": {"chi2": chi2, "N": len(entries), "fit_rd": fit_dict},
        "SNe_stat_only": sne_dict
    }
    print(json.dumps(out, indent=2))

    if args.plot:
        import matplotlib.pyplot as plt
        z_grid = np.linspace(0.0, 2.5, 300)
        Ez_grid = np.array([Ez(z) for z in z_grid])
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(z_grid, Ez_grid)
        ax.set_xlabel("z"); ax.set_ylabel("E(z)"); ax.set_title(f"E(z) [sfv={args.sfv_clock}, eps={args.eps}]")
        fig.tight_layout()
        fig.savefig("laneB_SFV_Ez.png", dpi=160)
        plt.close(fig)

if __name__ == "__main__":
    main()
