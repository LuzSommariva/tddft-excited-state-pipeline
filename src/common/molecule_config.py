"""
Per-molecule configuration for the Tier 1->3 PCM+TDDFT pipeline.

This does NOT replace or modify any existing Coumarin 102 script
(04_s0_optimize_pcm.py, 06_s1_optimize_gpu_pcm.py, tddft_ase_calculator.py,
11_s1_md_run.py, 12_sample_trajectory.py, 13_ensemble_emission_pcm.py) -- those keep
their own hardcoded constants exactly as validated and run; nothing about the
completed Coumarin 102 result depends on this file. It exists so that the
p-terphenyl pipeline (and any molecule after it) can reuse the same xc/basis/PCM
conventions already validated on Coumarin 102, looked up by name instead of
copy-pasted and silently drifting.

XC and BASIS are shared across molecules on purpose: the whole point of testing a
second molecule is to validate the SAME methodology, not a different one, unless a
specific change is explicitly proposed and confirmed (see README.md decision
records).
"""

XC = "wb97x-d3bj"
BASIS = "6-31g*"

MOLECULES = {
    "coumarin102": dict(
        name="Coumarin 102 (C102)",
        smiles="CC1=CC(=O)OC2=C1C=C3CCCN4C3=C2CCC4",
        solvent_method="IEF-PCM",
        solvent_eps=24.55,
        solvent_name="ethanol",
        # Well-documented absorption max for this laser dye; consistently reported in the
        # ~385-395 nm range in ethanol across sources -- 390 nm used as the representative
        # literature point (added 2026-08-26, Task 1 revision item 2).
        exp_absorption_nm=390.0,
        exp_emission_nm=468.0,
        exp_emission_min_nm=466.0,
        exp_emission_max_nm=470.0,
        exp_reference="Literature range 466-470 nm, ethanol (already used in "
                       "10_broaden_and_plot.py / 14_plot_emission_final.py)",
    ),
    "p_terphenyl": dict(
        name="p-terphenyl",
        smiles="c1ccc(cc1)-c1ccc(cc1)-c1ccccc1",
        solvent_method="IEF-PCM",
        solvent_eps=2.02,
        solvent_name="cyclohexane",
        exp_absorption_nm=276.0,
        exp_emission_nm=338.0,
        # Single literature point, not a range -- min==max on purpose, do not pad
        # this into a fabricated band.
        exp_emission_min_nm=338.0,
        exp_emission_max_nm=338.0,
        exp_reference="Berlman, 1971 (cyclohexane)",
    ),
}


def get(molecule_key):
    return MOLECULES[molecule_key]
