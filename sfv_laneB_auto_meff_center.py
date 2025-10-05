
#!/usr/bin/env python3
"""
sfv_laneB_auto_meff_center.py

Model-driven Lane-B exporter:
Centers the early-time injection using the dynamical onset rule
    m_eff(r_LHM) = 3 H(z_c),
where r_LHM is the rising half-maximum of the smoothed bounce energy bump.
Builds alpha(z), E2(z) and an effective-fluid table from the bounce profile.

Usage (example):
    export LAMBDA_STAR_EV=0.72
    python sfv_laneB_auto_meff_center.py \
        --bg background_profile.csv \
        --gr goldenRunDetails_v4f_more3.py \
        --pt_json p_t_meff.json \
        --h 0.72 --Omega_m 0.322 --omega_b 0.0224 \
        --V_to_rhoCrit0 6.404944382e+09 \
        --v_phys_planck 4.2e-5 \
        --sigma_ln1pz 0.25 \
        --out alpha_table_laneB_auto_meff_L072.csv
"""

import sys, os, importlib.util, json, argparse
import numpy as np, pandas as pd
from math import pi

# --- constants ---
c_kms   = 299792.458
T_CMB   = 2.7255
N_eff   = 3.046
omega_gamma = 2.469e-5 * (T_CMB/2.7255)**4
Mpl_eV  = 2.435e27  # reduced Planck mass
hbar    = 6.582119569e-16  # eV*s

def smooth_boxcar(y, win=11):
    win = max(3, int(win))
    if win % 2 == 0: win += 1
    pad = win//2
    yp = np.pad(y, (pad,pad), mode="edge")
    kernel = np.ones(win)/win
    ys = np.convolve(yp, kernel, mode="valid")
    return ys

def load_module(py_path, name="golden"):
    spec = importlib.util.spec_from_file_location(name, py_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def fwhm_LR(x, y):
    """Return indices (L, R, i_pk) for the FWHM and the peak index."""
    i = int(np.argmax(y))
    ypk = y[i]; half = 0.5*ypk
    # left crossing
    L = i
    while L > 0 and y[L] > half:
        L -= 1
    # right crossing
    R = i
    n = len(y)
    while R < n-1 and y[R] > half:
        R += 1
    return L, R, i

def main():
    ap = argparse.ArgumentParser(description="Lane-B exporter with model-driven (m_eff=3H) centering.")
    ap.add_argument("--bg", required=True, help="background_profile.csv (columns: r, Phi, phi)")
    ap.add_argument("--gr", required=True, help="goldenRunDetails_v4f_more3.py (must define potential_rescaled)")
    ap.add_argument("--pt_json", default="", help="JSON with rescaled params; else reads p_t from the module")
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--Omega_m", type=float, required=True)
    ap.add_argument("--omega_b", type=float, required=True)
    ap.add_argument("--V_to_rhoCrit0", type=float, required=True, help="Λ*^4 / ρ_crit0(h)")
    ap.add_argument("--v_phys_planck", type=float, default=4.2e-5, help="v_phys in M_pl units (default 4.2e-5)")
    ap.add_argument("--sigma_ln1pz", type=float, default=0.25, help="Width in ln(1+z) mapped from bounce FWHM")
    ap.add_argument("--out", default="alpha_table_laneB_auto_meff.csv")
    args = ap.parse_args()

    # cosmology
    h  = args.h
    H0 = 100.0*h
    h2 = h*h
    Or = (omega_gamma * (1.0 + 0.2271 * N_eff)) / h2
    Ode= 1.0 - args.Omega_m - Or
    Ob = args.omega_b / h2
    Og = omega_gamma / h2

    # load background
    bg = pd.read_csv(args.bg)
    r   = bg["r"].to_numpy()
    Phi = bg["Phi"].to_numpy()
    phi = bg["phi"].to_numpy()

    # load potential
    mod = load_module(args.gr, "golden")
    if not hasattr(mod, "potential_rescaled"):
        raise RuntimeError("potential_rescaled not found in goldenRun module")
    Vfun = mod.potential_rescaled
    if args.pt_json:
        p = json.load(open(args.pt_json, "r"))
    else:
        p = getattr(mod, "p_t", None)
    if p is None:
        raise RuntimeError("No p_t parameter dict provided/found.")

    lam_phi = p.get("lam_phi", 0.1)
    v_phi_t = p.get("v_phi_t", 1.0)
    bias_t  = p.get("bias_t", 1.01)
    lam_t   = p.get("lam_t", 0.1)
    mu2_t   = p.get("mu2_t", 0.05)
    g_portal= p.get("g_portal_t", 0.1)

    # helper: second derivatives (rescaled)
    def V_PP(P, ps):
        return lam_phi*(3*P*P - v_phi_t*v_phi_t) + 2*bias_t + 2*g_portal*(ps*ps)
    def V_pp(P, ps):
        return lam_t*(3*ps*ps - 1.0) - 2*mu2_t + 2*g_portal*(P*P)
    def V_Pp(P, ps):
        return 4*g_portal*P*ps

    # build ΔV along the path
    V_arr = np.array([Vfun(Phi[i], phi[i], p) for i in range(len(r))], dtype=float)
    V_false = float(np.mean(V_arr[-max(5, len(V_arr)//50):]))
    dV = np.maximum(V_arr - V_false, 0.0)
    dV_s = smooth_boxcar(dV, win=max(5, len(dV)//200))

    # FWHM and locate rising half-maximum (onset)
    L, R, i_pk = fwhm_LR(r, dV_s)
    r_L = r[L]
    FWHM_r = max(r[R] - r[L], 1e-12)
    std_r  = FWHM_r / (2.0*np.sqrt(2.0*np.log(2.0)))

    # mass matrix at rising half-maximum
    P_L, p_L = Phi[L], phi[L]
    M11 = V_PP(P_L, p_L); M22 = V_pp(P_L, p_L); M12 = V_Pp(P_L, p_L)
    tr = M11 + M22; det = M11*M22 - M12*M12
    disc = max(tr*tr/4.0 - det, 0.0)
    lam_min = tr/2.0 - np.sqrt(disc)
    lam_min = max(lam_min, 0.0)

    # convert to physical mass: m_eff = (Λ*^2 / v_phys) * sqrt(lam_min)
    v_phys_eV   = args.v_phys_planck * Mpl_eV
    Lambda_star = float(os.environ.get("LAMBDA_STAR_EV", "0.72"))
    m_eff_phys  = (Lambda_star**2 / v_phys_eV) * np.sqrt(lam_min)  # eV

    # H(z) in eV
    H0_s = 100.0*h * 1000.0 / (3.0856775814913673e22)
    H0_eV = H0_s * hbar
    def E2_base(z): return Or*(1+z)**4 + args.Omega_m*(1+z)**3 + Ode
    def H_eV(z):    return H0_eV * np.sqrt(E2_base(z))

    # solve 3 H(z_c) = m_eff_phys (binary search in z)
    target = m_eff_phys / 3.0
    zlo, zhi = 10.0, 2.0e5
    for _ in range(120):
        zmid = np.sqrt(zlo*zhi)
        if H_eV(zmid) < target: zlo = zmid
        else: zhi = zmid
    z_c = 0.5*(zlo+zhi)

    # map r -> z with Gaussian width set by std_r
    k = args.sigma_ln1pz / max(std_r, 1e-12)
    ln1pz = k*(r - r_L) + np.log1p(z_c)
    z_map = np.expm1(ln1pz)

    # window ΔV to avoid long tails
    tau = 1e-3
    dV_peak = float(np.max(dV_s)) if len(dV_s)>0 else 0.0
    dV_w = np.where(dV_s >= tau*dV_peak, dV_s, 0.0)

    # build Ω_X(z)
    OmegaX_core = dV_w * args.V_to_rhoCrit0

    # pad in z-domain
    z_min, z_max = z_map.min(), z_map.max()
    z_lo = np.geomspace(1e-4, max(1e-3, 0.5*z_min), 400)
    z_hi = np.geomspace(min(5e4, 1.1*z_max), 1.0e5, 600)
    z = np.concatenate([z_lo, z_map, z_hi])
    OmegaX = np.concatenate([np.zeros_like(z_lo), OmegaX_core, np.zeros_like(z_hi)])

    order = np.argsort(z)
    z = z[order]; OmegaX = OmegaX[order]
    z, uniq = np.unique(z, return_index=True)
    OmegaX = OmegaX[uniq]

    # alpha and background
    E2b   = E2_base(z)
    alpha = np.sqrt(1.0 + OmegaX/np.maximum(E2b, 1e-60))
    E2m   = (alpha**2) * E2b

    # effective fluid diagnostics
    a = 1.0/(1.0+z)
    ln_a = np.log(a)
    ln_rho = np.log(np.maximum((alpha**2 - 1.0)*E2b, 1e-120))
    dlnrho_dlna = np.gradient(ln_rho, ln_a)
    w_eff = -1.0 - (1.0/3.0)*dlnrho_dlna

    tab = pd.DataFrame({
        "z": z, "a": a, "alpha": alpha,
        "E2_base": E2b, "E2_mod": E2m,
        "Omega_X_of_z": (alpha**2 - 1.0)*E2b,
        "w_eff_of_z": w_eff
    })
    tab.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(tab)} rows.  z_c (from m_eff=3H) ≈ {z_c:.2f}")

    # Quick anchors for visibility
    def E_mod_interp(zq):
        x  = np.log1p(z)
        y  = np.sqrt(E2m)
        xq = np.clip(np.log1p(zq), x.min(), x.max())
        return np.interp(xq, x, y)

    def z_star_fit(omega_b, omega_m):
        g1 = 0.0783 * (omega_b)**(-0.238) / (1 + 39.5*(omega_b)**0.763)
        g2 = 0.560 / (1 + 21.1*(omega_b)**1.81)
        return 1048 * (1 + 0.00124 * (omega_b)**(-0.738)) * (1 + g1 * (omega_m)**g2)

    def z_drag_fit(omega_b, omega_m):
        b1 = 0.313 * (omega_m)**(-0.419) * (1 + 0.607 * (omega_m)**0.674)
        b2 = 0.238 * (omega_m)**0.223
        return 1291 * (omega_m**0.251) / (1 + 0.659 * (omega_m**0.828)) * (1 + b1 * (omega_b**b2))

    def rs_integral(z_from, z_to=1e5, nz=60000):
        zz = np.linspace(z_from, z_to, nz)
        E  = E_mod_interp(zz)
        R  = (3.0/4.0) * (Ob/Og) / (1.0+zz)
        cs = c_kms/np.sqrt(3*(1+R))
        Hz = H0*E
        return np.trapz(cs/Hz, zz)

    def DM(zup, nz=30000):
        zz = np.linspace(0.0, zup, nz)
        E  = E_mod_interp(zz)
        return (c_kms/H0)*np.trapz(1.0/E, zz)

    omega_m = h2*args.Omega_m
    zstar = z_star_fit(args.omega_b, omega_m)
    zdrag = z_drag_fit(args.omega_b, omega_m)
    rs_star = rs_integral(zstar)
    rd      = rs_integral(zdrag)
    DM_star = DM(zstar)
    theta_star = rs_star/DM_star
    lA  = pi/theta_star
    print(f"Anchors(pred): r_s*={rs_star:.3f} Mpc, r_d={rd:.3f} Mpc, lA={lA:.3f}")

if __name__ == "__main__":
    main()
