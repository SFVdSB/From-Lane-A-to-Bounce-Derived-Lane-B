# Simple Makefile to reproduce GAUSS run and export alpha(z)
.PHONY: all gauss meff

all: gauss meff

gauss:
	python laneB_with_SFVclock.py --H0 72.0 --Om 0.322 --Obh2 0.0224 \	  --sfv-clock gauss --eps 0.534 --gauss-zc 15500 --gauss-sig 0.30 \	  --bao_mean desi_2024_gaussian_bao_ALL_GCcomb_mean.txt \	  --bao_cov  desi_2024_gaussian_bao_ALL_GCcomb_cov.txt \	  --use-sne --sne sne_pantheonplus_relative_STAT.csv --fit-rd

meff:
	python sfv_laneB_auto_meff_center.py --bg background_profile.csv \	  --gr goldenRunDetails_v4f_more3.py --pt_json p_t_template.json \	  --h 0.72 --Omega_m 0.322 --omega_b 0.0224 \	  --V_to_rhoCrit0 6.404944382e+09 --v_phys_planck 4.2e-5 \	  --sigma_ln1pz 0.25 --out alpha_table_laneB_meff_zc15500.csv
