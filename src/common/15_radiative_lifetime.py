"""
Post-processing only -- Radiative (fluorescence) lifetime from existing (E, f) data

Reads any of this pipeline's ensemble-emission CSVs (Tier 2 or Tier 3, any molecule --
column detection is name-based, same convention as 14_plot_emission_final.py) and
computes, PER ROW, the spontaneous-emission Einstein A coefficient and the
corresponding purely-radiative lifetime, from the already-computed vertical emission
energy and oscillator strength alone. No new SCF/TDDFT/MD is run or required -- this
is a closed-form calculation on numbers this pipeline has already produced.

FORMULA (derived and cross-validated two independent ways before use -- see
derivation notes below; both give the exact same numeric result):

    A (s^-1) = (2 * pi * e^2) / (m_e * c * eps0 * lambda^2) * f
             = 8 * pi^2 * r_e * c * f / lambda^2          [r_e = classical electron radius]
             = 2 * alpha^3 * E(Hartree)^2 * f / t_au       [equivalent, atomic units]
             = 4.339197e7 * E(eV)^2 * f                    [s^-1, convenience closed form]

    tau_rad (ns) = 1e9 / A = 23.0459 / (E(eV)^2 * f)

This is the standard, non-degenerate (g_i = g_k = 1) electric-dipole spontaneous
emission rate expressed via oscillator strength -- appropriate here because a closed-
shell singlet ground state and its lowest singlet excited state (S1, what every script
in this pipeline reports) are both non-degenerate. Source: this exact A-vs-(f, lambda)
relation (via the classical electron radius r_e) is the standard form used across
atomic/molecular radiative-rate literature (e.g. Rybicki & Lightman-style derivations);
independently re-derived here from the atomic-units QED spontaneous-emission rate
A(a.u.) = (4/3)*omega^3*|mu|^2*alpha^3 combined with the standard oscillator-strength
definition f = (2/3)*omega*|mu|^2 (both in Hartree/a.u., g=1) -- the two derivations
were checked numerically against each other (and against the exact CODATA value of the
classical electron radius) before this script was written; they agree to all digits
shown. An earlier from-memory attempt at the same formula in "textbook SI" form
(A = 2*e^2*omega^2 / (3*eps0*m_e*c^3) * f) was off by a factor of ~4pi/3 -- a classic
4*pi*eps0-vs-eps0 (SI vs. Gaussian) slip -- and was discarded in favor of the two
independently-agreeing derivations used here. If this number is ever cross-checked
against another source, check the eps0-vs-4*pi*eps0 convention first.

IMPORTANT SCOPE NOTE -- what this script does NOT do:
  - No solvent local-field / refractive-index correction (Strickler-Berg's optional
    n^2 factor, or an Onsager-type n*(n^2+2)^2/9 local-field factor) is applied. This
    is the bare/vacuum formula. PCM already relaxes the electronic structure (and
    hence E and f) to the solvent's dielectric response; whether an *additional*
    refractive-index rate correction on top of that is double-counting or a genuinely
    separate (missing) physical effect is a real, unresolved modeling choice, not
    settled here. For reference only, since it's NOT applied: n(ethanol) ~= 1.361,
    n(cyclohexane) ~= 1.426 -- an n^2 correction would multiply every A (and divide
    every tau) below by ~1.85 (Coumarin 102/ethanol) or ~2.03 (p-terphenyl/cyclohexane)
    if applied. Use --refractive-index-n2 to apply it explicitly (off by default).
  - No degeneracy factor beyond g=1 (appropriate for S1<->S0 in a closed-shell singlet
    system; would need revisiting for a system with a degenerate excited state).
  - This computes the PURELY RADIATIVE lifetime implied by the TDDFT transition dipole
    alone. It is not directly the same as an experimentally OBSERVED fluorescence
    lifetime unless the fluorescence quantum yield is 1 (tau_observed =
    tau_radiative * Phi_fluorescence) -- nonradiative decay pathways (internal
    conversion, intersystem crossing) are not, and cannot be, captured by this
    calculation; they require entirely different information (e.g. Phi from
    experiment, or explicit nonadiabatic-dynamics/ISC rate calculations).

ENSEMBLE AVERAGING -- two numbers are reported, not one, because they are not the same
and the difference matters:
  - "mean-of-A" (reported as the primary/physically-appropriate ensemble estimate):
    average the per-snapshot rate A_i, i.e. <A> = (1/N) sum_i A_i, then
    tau_ensemble = 1/<A>. This is the physically consistent choice: each sampled
    geometry represents a real (sub-)population decaying at its own instantaneous
    rate, and rates (not lifetimes) are what add/average linearly across a
    population.
  - "A-of-means" (reported for reference/comparison only): plug the ensemble-averaged
    <E> and <f> into the same formula, A(<E>,<f>). This is what a hasty single-number
    calculation would give and is NOT recommended as the headline result, since A
    depends on E^2 (nonlinear) -- Jensen's inequality means these two generally differ
    (A-of-means slightly overweights the spread in E) -- but it's reported alongside
    for transparency, and to make that gap visible rather than hiding it.

Usage:
  python3 15_radiative_lifetime.py --csv ensemble_emission_pcm_tier3.csv \\
      --label "Coumarin 102, Tier 3"
  python3 15_radiative_lifetime.py --csv ensemble_emission.csv \\
      --label "Coumarin 102, Tier 2" --refractive-index-n2 1.361
"""
import argparse
import csv
import math

E_TO_A_COEFF_S_INV_PER_EV2 = 4.339197e7  # A(s^-1) = COEFF * E(eV)^2 * f  (see docstring)


def load_csv(path):
    energies_ev, oscs = [], []
    with open(path) as fh:
        reader = csv.DictReader(fh)
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
    return energies_ev, oscs, energy_col, osc_col


def A_coeff(E_eV, f, n2=None):
    """Einstein A (s^-1) from vertical emission energy (eV) and oscillator strength."""
    A = E_TO_A_COEFF_S_INV_PER_EV2 * (E_eV ** 2) * f
    if n2 is not None:
        A *= n2
    return A


def main():
    energies_ev, oscs, energy_col, osc_col = load_csv(args.csv)
    n = len(energies_ev)
    print(f"[info] {n} rows loaded from {args.csv}")
    print(f"[info] columns used: energy='{energy_col}' (eV), oscillator strength='{osc_col}'")
    if n == 0:
        raise SystemExit(f"[ERROR] {args.csv} has no data rows.")

    n2 = None
    if args.refractive_index_n2 is not None:
        n2 = args.refractive_index_n2 ** 2
        print(f"[info] applying refractive-index correction: n={args.refractive_index_n2} "
              f"(n^2={n2:.4f}) -- NOT applied by default, see docstring")

    per_row_A = [A_coeff(E, f, n2) for E, f in zip(energies_ev, oscs)]
    per_row_tau_ns = [1e9 / A for A in per_row_A]

    mean_A = sum(per_row_A) / n
    tau_from_mean_A_ns = 1e9 / mean_A

    mean_E = sum(energies_ev) / n
    mean_f = sum(oscs) / n
    A_of_means = A_coeff(mean_E, mean_f, n2)
    tau_of_means_ns = 1e9 / A_of_means

    print()
    print(f"[RESULT] {args.label or args.csv}")
    print(f"[RESULT]   N = {n} sampled geometries")
    print(f"[RESULT]   <E> = {mean_E:.4f} eV, <f> = {mean_f:.4f}")
    print(f"[RESULT]   per-sample A range: {min(per_row_A):.4e} - {max(per_row_A):.4e} s^-1")
    print(f"[RESULT]   per-sample tau_rad range: {min(per_row_tau_ns):.4f} - "
          f"{max(per_row_tau_ns):.4f} ns")
    print(f"[RESULT]   mean-of-A  : <A> = {mean_A:.4e} s^-1  -> tau_rad = "
          f"{tau_from_mean_A_ns:.4f} ns   *** primary/recommended ***")
    print(f"[RESULT]   A-of-means : A(<E>,<f>) = {A_of_means:.4e} s^-1  -> tau_rad = "
          f"{tau_of_means_ns:.4f} ns   (reference only, see docstring)")
    print(f"[RESULT]   difference: {100*(tau_of_means_ns - tau_from_mean_A_ns) / tau_from_mean_A_ns:+.2f}% "
          f"(A-of-means vs. mean-of-A)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label", default=None,
                         help="Human-readable label for the [RESULT] header (e.g. "
                              "'Coumarin 102, Tier 3')")
    parser.add_argument("--refractive-index-n2", type=float, default=None,
                         help="If given, apply an n^2 local-field/Strickler-Berg-style "
                              "correction using this refractive index n (off by "
                              "default -- see docstring's SCOPE NOTE)")
    args = parser.parse_args()
    main()
