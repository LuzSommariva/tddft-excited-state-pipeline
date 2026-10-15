"""
Tier 3, final plot -- Emission spectrum with Gaussian broadening -- Coumarin 102 (C102)

Reads the emission CSV produced by 13_ensemble_emission_pcm.py (one row per MD-sampled
geometry: vertical S1->S0 emission energy + oscillator strength, computed with PCM
(IEF-PCM, ethanol, eps=24.55, equilibrium_solvation=True) + full TDDFT + 6-31g*, on
geometries drawn from the real S1 excited-state MD trajectory) and builds the broadened
"stick spectrum":

    I(E) = sum_i  w_i * exp( -(E - E_i)^2 / (2 * sigma^2) )

Read-only / plotting only: this script does not touch any file belonging to the Tier 3
run (CSV, XYZ samples, MD trajectory, checkpoints, resume markers) -- it only reads the
CSV given by --csv and writes the two output plot files (PNG + PDF).

Column detection is name-based (case-insensitive substring match: "energy"+"ev" for the
energy column, "osc" for oscillator strength) rather than hardcoded to a fixed schema,
so this script also runs unchanged on the Tier 2 ensemble_emission.csv or other CSV
variants from this pipeline, as long as they carry an eV energy column and an oscillator
strength column.

Weighting (--weight-e2f, default off = plain f): see 10_broaden_and_plot.py's docstring
for the derivation of E^2*f as the more physically correct weight for an EMISSION band
(Einstein A coefficient ~ omega^2 * f), vs. plain f which is the more common convention
in NEA absorption papers.

Experimental reference: looked up from molecule_config.py via --molecule (default
"coumarin102", preserving this script's original behavior -- ~466-470 nm, ethanol).
This lookup is ONLY correct if --csv actually belongs to the molecule named by
--molecule -- verify that against the run's own log/atom-count before trusting it
blindly, the same way the Coumarin 102 vs. PNA smoke-test ambiguity was resolved for
this CSV on 2026-08-23. Override --exp-ref-min-nm/--exp-ref-max-nm or pass
--no-exp-ref for a molecule not in molecule_config.py.

Usage:
  python3 14_plot_emission_final.py
  python3 14_plot_emission_final.py --weight-e2f --sigma-ev 0.10
  python3 14_plot_emission_final.py --csv ensemble_emission.csv --no-exp-ref
  python3 14_plot_emission_final.py --molecule p_terphenyl \\
      --csv ensemble_emission_pcm_tier3_pterphenyl.csv \\
      --out-prefix emission_spectrum_tier3_pterphenyl_final
"""
import argparse
import csv as csv_module

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from molecule_config import XC, BASIS, MOLECULES

HC_EV_NM = 1239.84  # eV*nm, E(eV) = HC_EV_NM / lambda(nm)


def load_csv(path):
    energies_ev, oscs = [], []
    with open(path) as fh:
        reader = csv_module.DictReader(fh)
        fieldnames = reader.fieldnames or []
        energy_col = next((c for c in fieldnames
                            if "energy" in c.lower() and "ev" in c.lower()), None)
        osc_col = next((c for c in fieldnames if "osc" in c.lower()), None)
        if energy_col is None or osc_col is None:
            raise SystemExit(
                f"[ERROR] Could not find an energy (eV) and/or oscillator-strength "
                f"column in {path}. Columns found: {fieldnames}"
            )
        for row in reader:
            energies_ev.append(float(row[energy_col]))
            oscs.append(float(row[osc_col]))
    return np.array(energies_ev), np.array(oscs), energy_col, osc_col


def build_broadened_spectrum(energies_ev, weights, sigma_ev, npoints):
    e_grid_ev = np.linspace(energies_ev.min() - 5 * sigma_ev,
                             energies_ev.max() + 5 * sigma_ev,
                             npoints)
    intensity = np.zeros_like(e_grid_ev)
    for e_i, w_i in zip(energies_ev, weights):
        intensity += w_i * np.exp(-((e_grid_ev - e_i) ** 2) / (2 * sigma_ev ** 2))
    return e_grid_ev, intensity


def apply_modern_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.edgecolor": "#4A4A4A",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#DADADA",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def main():
    cfg = MOLECULES[args.molecule]
    energies_ev, oscs, energy_col, osc_col = load_csv(args.csv)
    n = len(energies_ev)
    print(f"[info] {n} geometries loaded from {args.csv}")
    print(f"[info] columns used: energy='{energy_col}' (eV), oscillator strength='{osc_col}'")
    if n == 0:
        raise SystemExit(f"[ERROR] {args.csv} has no data rows.")

    if args.weight_e2f:
        weights = (energies_ev ** 2) * oscs
        weight_label = "E^2 * f  (emission-rate weighted)"
    else:
        weights = oscs
        weight_label = "f  (oscillator strength)"

    e_grid_ev, intensity = build_broadened_spectrum(
        energies_ev, weights, args.sigma_ev, args.npoints
    )
    intensity = intensity / intensity.max()  # normalize to unit peak
    wavelength_grid_nm = HC_EV_NM / e_grid_ev

    peak_idx = int(np.argmax(intensity))
    peak_nm = wavelength_grid_nm[peak_idx]
    peak_ev = e_grid_ev[peak_idx]

    sample_wavelengths_nm = HC_EV_NM / energies_ev

    use_exp_ref = not args.no_exp_ref
    exp_min_nm = args.exp_ref_min_nm if args.exp_ref_min_nm is not None else cfg["exp_emission_min_nm"]
    exp_max_nm = args.exp_ref_max_nm if args.exp_ref_max_nm is not None else cfg["exp_emission_max_nm"]
    exp_mid_nm = 0.5 * (exp_min_nm + exp_max_nm) if use_exp_ref else None
    exp_is_point = use_exp_ref and exp_min_nm == exp_max_nm

    # --- console summary -----------------------------------------------------------
    print(f"[info] weighting scheme: {weight_label}")
    print(f"[info] Gaussian broadening sigma = {args.sigma_ev} eV")
    print(f"[info] energy range (raw samples): {energies_ev.min():.3f} - "
          f"{energies_ev.max():.3f} eV "
          f"({sample_wavelengths_nm.min():.1f} - {sample_wavelengths_nm.max():.1f} nm)")
    print(f"[info] oscillator strength range: {oscs.min():.4f} - {oscs.max():.4f}")
    print(f"[RESULT] Calculated emission peak (broadened spectrum): "
          f"{peak_nm:.1f} nm ({peak_ev:.3f} eV)")
    if use_exp_ref:
        ref_str = f"{exp_min_nm:.0f} nm" if exp_is_point else f"{exp_min_nm:.0f}-{exp_max_nm:.0f} nm"
        print(f"[RESULT] Experimental reference ({cfg['name']} / {cfg['solvent_name']}): "
              f"{ref_str} ({cfg['exp_reference']})")
        print(f"[RESULT] Offset (calculated - experimental): {peak_nm - exp_mid_nm:+.1f} nm")
    else:
        print("[info] no experimental reference plotted (--no-exp-ref)")

    # --- figure ----------------------------------------------------------------------
    apply_modern_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.fill_between(wavelength_grid_nm, intensity, color="#1B6E7A", alpha=0.18, lw=0)
    ax.plot(wavelength_grid_nm, intensity, color="#0B4F58", lw=2.4,
             label=f"Computed emission spectrum (N={n}, σ={args.sigma_ev} eV)")

    # rug of the raw MD-sampled geometries, to show this is an ensemble, not one point
    rug_y = -0.045
    ax.plot(sample_wavelengths_nm, np.full_like(sample_wavelengths_nm, rug_y),
            "|", color="#0B4F58", markersize=14, markeredgewidth=1.6, alpha=0.75,
            label="Individual MD-sampled geometries")

    solvent_label = cfg["solvent_name"].capitalize()
    if use_exp_ref and exp_is_point:
        ax.axvline(exp_min_nm, color="#E8912D", lw=2.0,
                   label=f"Experimental ref. ({solvent_label}, {exp_min_nm:.0f} nm)")
    elif use_exp_ref:
        ax.axvspan(exp_min_nm, exp_max_nm, color="#E8912D", alpha=0.22, lw=0,
                   label=f"Experimental ref. ({solvent_label}, "
                         f"{exp_min_nm:.0f}-{exp_max_nm:.0f} nm)")

    ax.axvline(peak_nm, color="#B23A48", ls="--", lw=1.6,
               label=f"Calculated peak ({peak_nm:.1f} nm)")

    ax.set_ylim(rug_y - 0.03, 1.12)
    ax.set_xlim(max(wavelength_grid_nm.min(), 250), min(wavelength_grid_nm.max(), 700))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized emission intensity")
    xc_tag = XC.replace("wb97x-d3bj", "ωB97X-D3BJ")
    ax.set_title(f"{xc_tag}/{BASIS.upper()} · IEF-PCM {cfg['solvent_name']} "
                 f"(ε={cfg['solvent_eps']}) · weight: {weight_label}",
                 fontsize=9.5, color="#555555", fontweight="normal", pad=10)
    fig.suptitle(f"{cfg['name']} -- TDDFT/PCM Emission Spectrum (Tier 3, MD ensemble)",
                 fontsize=15, fontweight="bold", y=0.985)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = f"{args.out_prefix}.png"
    pdf_path = f"{args.out_prefix}.pdf"
    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path)
    print(f"[DONE] Saved {png_path} (dpi={args.dpi}) and {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--molecule", choices=sorted(MOLECULES), default="coumarin102",
                         help="Looked up in molecule_config.py for title, solvent, and "
                              "experimental reference -- must match what --csv actually "
                              "contains, this script does not verify that for you")
    parser.add_argument("--csv", default="ensemble_emission_pcm_tier3.csv")
    parser.add_argument("--out-prefix", default="emission_spectrum_tier3_final",
                         help="Output files are <out-prefix>.png and <out-prefix>.pdf")
    parser.add_argument("--sigma-ev", type=float, default=0.15,
                         help="Gaussian broadening width (std. dev.), in eV")
    parser.add_argument("--npoints", type=int, default=4000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--weight-e2f", action="store_true",
                         help="Weight each geometry by E_i^2*f_i instead of f_i alone "
                              "(closer to the true emission rate; see docstring)")
    parser.add_argument("--exp-ref-min-nm", type=float, default=None,
                         help="Overrides molecule_config.py's value for --molecule")
    parser.add_argument("--exp-ref-max-nm", type=float, default=None,
                         help="Overrides molecule_config.py's value for --molecule")
    parser.add_argument("--no-exp-ref", action="store_true",
                         help="Do not plot/report an experimental reference band "
                              "(use this if --csv is not a Coumarin 102 result)")
    args = parser.parse_args()
    main()
