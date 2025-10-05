# From Lane A to Bounce‑Derived Lane B: BAO & $H_0$ with Emergent Time in the SFV/dSB Model

This repository contains a fully reproducible pipeline that connects the **SFV/dSB bounce** to **pre-recombination expansion** (“Lane B”) that matches **DESI BAO** at **H0 ≈ 72 km/s/Mpc** while leaving recombination geometry intact.

## Contents
- `laneB_with_SFVclock.py` — GAUSS–clock driver for Lane A/B phenomenology; computes BAO (DESI GCcomb) and optional SNe (Pantheon+ stat-only).
- `sfv_laneB_auto_meff_center.py` — meff–centered exporter that reads the golden-run background and builds an `alpha(z)` table (Lane B).
- `goldenRunDetails_v4f_more3.py`, `p_t_template.json`, `background_profile.csv` — bounce background and potential template.
- `desi_2024_gaussian_bao_ALL_GCcomb_mean.txt`, `desi_2024_gaussian_bao_ALL_GCcomb_cov.txt` — DESI BAO inputs.
- `alpha_table_laneB_meff_zc15500.csv` — example alpha(z) table (Lane B exporter) used in the paper.
- `sfv_gauss_rd13991_fit.json` — JSON summary of the tuned GAUSS clock run.

## Environment
- Python 3.11, NumPy, SciPy, Pandas, Matplotlib.
- See `requirements.txt` or `environment.yml` for exact versions.

## Reproduction (one-liners)

### (A) Lane A/B GAUSS clock (target rd=139.91 at H0=72, Om=0.322)
```bash
python laneB_with_SFVclock.py --H0 72.0 --Om 0.322 --Obh2 0.0224 --sfv-clock gauss --eps 0.531 --gauss-zc 15460 --gauss-sig 0.30 --bao_mean desi_2024_gaussian_bao_ALL_GCcomb_mean.txt --bao_cov  desi_2024_gaussian_bao_ALL_GCcomb_cov.txt --use-sne --sne sne_pantheonplus_relative_STAT.csv --fit-rd
```

### (B) Lane B meff-centered exporter (amplitude anchored at Λ* = 0.72 eV)

```bash
python sfv_laneB_auto_meff_center.py   --bg background_profile.csv   --gr goldenRunDetails_v4f_more3.py   --pt_json p_t_template.json   --h 0.72 --Omega_m 0.322 --omega_b 0.0224   --V_to_rhoCrit0 6.404944382e+09   --v_phys_planck 4.2e-5 --sigma_ln1pz 0.25   --out alpha_table_laneB_meff_zc15500.csv
```

## Notes
- BAO χ² is computed with DESI GCcomb full covariance (N=12). The script can also report a nuisance `rd`-rescaling best-fit (`--fit-rd`) for reference.
- The GAUSS clock parameters above are **bounce-anchored**: the amplitude normalization comes from Λ* = 0.72 eV; only the onset timing is adjusted consistent with the `m_eff = 3H(z_c)` crossing rule.
- The clock is exponentially suppressed at recombination and at BBN redshifts.

## License
MIT
