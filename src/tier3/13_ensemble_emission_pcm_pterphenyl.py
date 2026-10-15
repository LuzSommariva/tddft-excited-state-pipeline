"""
Tier 3, paso 4 -- Emision vertical sobre las muestras de la trayectoria de MD, con PCM
-- p-terphenyl

Mismo procedimiento que 13_ensemble_emission_pcm.py (Cumarina 102), parametrizado
via molecule_config.py. Default --samples/--out con nombres DISTINTOS a los de C102
a proposito (ver 11_s1_md_run_pterphenyl.py sobre por que evitar colisiones de
nombre de archivo entre las dos corridas es necesario, no cosmetico -- este script
ademas RETOMA desde las filas ya escritas si --out ya existe, asi que un nombre
compartido con el CSV de C102 haria que crea que ya termino sin escribir nada
nuevo).

Uso:
  python3 13_ensemble_emission_pcm_pterphenyl.py --gpu
  (default: CPU, s1_md_samples_pterphenyl.xyz, nstates=5, checkpoint-every=5,
   out=ensemble_emission_pcm_tier3_pterphenyl.csv)
"""
import argparse
import csv
import os
import time
from pyscf import gto, dft

from molecule_config import XC, BASIS, MOLECULES

cfg = MOLECULES["p_terphenyl"]
SOLVENT_METHOD = cfg["solvent_method"]
SOLVENT_EPS = cfg["solvent_eps"]
STATE = 1  # S1: primer estado excitado
# excitation_state/nstates_computed/state_gap_next_eV/state{2..5}_*: mismo agregado que
# 13_ensemble_emission_pcm.py (revision 2026-08-26), a pedido de una revision del
# modelo -- ver ese script para el razonamiento completo.
CSV_FIELDS = ["sample_index", "energy_eV", "oscillator_strength", "wavelength_nm",
              "scf_tddft_time_s", "excitation_state", "nstates_computed",
              "state_gap_next_eV",
              "state2_energy_eV", "state2_osc", "state3_energy_eV", "state3_osc",
              "state4_energy_eV", "state4_osc", "state5_energy_eV", "state5_osc"]


def read_xyz_frames(path):
    frames = []
    with open(path) as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        natm = int(lines[i].strip())
        comment = lines[i + 1].rstrip("\n")
        atom_lines = lines[i + 2:i + 2 + natm]
        atoms = []
        for line in atom_lines:
            symb, x, y, z = line.split()
            atoms.append((symb, (float(x), float(y), float(z))))
        frames.append((comment, atoms))
        i += 2 + natm
    return frames


def n_completed_rows(csv_path):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path) as fh:
        return max(0, sum(1 for _ in fh) - 1)


def emission_at(atoms, gpu):
    mol = gto.M(atom=atoms, basis=BASIS, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = XC
    if gpu:
        mf = mf.to_gpu()
    mf = mf.PCM()
    mf.with_solvent.method = SOLVENT_METHOD
    mf.with_solvent.eps = SOLVENT_EPS
    mf.kernel()
    assert mf.converged, "SCF no convergio en una geometria de la ensemble"

    td = mf.TDDFT(equilibrium_solvation=True)
    td.nstates = args.nstates
    td.kernel()

    all_e_eV = td.e * 27.2114
    all_osc = td.oscillator_strength()
    e_eV = float(all_e_eV[STATE - 1])
    f_osc = float(all_osc[STATE - 1])
    if STATE < args.nstates:
        gap_next_eV = float(all_e_eV[STATE] - all_e_eV[STATE - 1])
    else:
        gap_next_eV = float("nan")
    other_states = []
    for i in range(1, 5):
        if i < args.nstates:
            other_states.extend([float(all_e_eV[i]), float(all_osc[i])])
        else:
            other_states.extend([float("nan"), float("nan")])
    return e_eV, f_osc, gap_next_eV, other_states


def main():
    frames = read_xyz_frames(args.samples)
    n_total = len(frames)
    print(f"[info] {n_total} geometrias cargadas de {args.samples}")
    print(f"[info] xc={XC}, basis={BASIS}, nstates={args.nstates}, state=S{STATE}")
    print(f"[info] PCM: method={SOLVENT_METHOD}, eps={SOLVENT_EPS} "
          f"({cfg['solvent_name']}), equilibrium_solvation=True")

    n_done = n_completed_rows(args.out)
    if n_done > 0:
        print(f"[info] {args.out} ya tiene {n_done} filas completas; retomando desde "
              f"la muestra {n_done}")
    else:
        with open(args.out, "w", newline="") as fh:
            csv.writer(fh).writerow(CSV_FIELDS)

    t0 = time.time()
    for idx in range(n_done, n_total):
        _, atoms = frames[idx]
        t_start = time.time()
        e_eV, f_osc, gap_next_eV, other_states = emission_at(atoms, args.gpu)
        dt = time.time() - t_start
        lambda_nm = 1239.84 / e_eV

        with open(args.out, "a", newline="") as fh:
            csv.writer(fh).writerow([idx, f"{e_eV:.6f}", f"{f_osc:.6f}",
                                      f"{lambda_nm:.2f}", f"{dt:.1f}",
                                      STATE, args.nstates, f"{gap_next_eV:.6f}"]
                                     + [f"{v:.6f}" for v in other_states])

        tag = "[GPU]" if args.gpu else ""
        print(f"[progress]{tag} muestra {idx+1}/{n_total}: E={e_eV:.3f} eV "
              f"({lambda_nm:.0f} nm), f={f_osc:.4f}, gap_S{STATE}->S{STATE+1}="
              f"{gap_next_eV:.3f} eV | {dt:.1f} s")

        if (idx + 1) % args.checkpoint_every == 0 or idx == n_total - 1:
            elapsed = time.time() - t0
            done_now = idx + 1 - n_done
            rate = elapsed / max(done_now, 1)
            remaining = (n_total - idx - 1) * rate
            print(f"[checkpoint] {idx+1}/{n_total} completadas | "
                  f"{elapsed:.0f} s transcurridos | ~{remaining:.0f} s restantes")

    print(f"[DONE] Ensemble completa: {n_total} geometrias en {args.out}")
    print(f"[timing] Tiempo total (esta corrida): {time.time()-t0:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="s1_md_samples_pterphenyl.xyz")
    parser.add_argument("--out", default="ensemble_emission_pcm_tier3_pterphenyl.csv")
    parser.add_argument("--nstates", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    main()
