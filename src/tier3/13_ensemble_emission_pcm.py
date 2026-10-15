"""
Tier 3, paso 4 -- Emision vertical sobre las muestras de la trayectoria de MD, con PCM
-- Cumarina 102 (C102)

Para cada geometria muestreada por 12_sample_trajectory.py: SCF + TDDFT completo bajo
PCM (IEF-PCM, etanol, eps=24.55, equilibrium_solvation=True), y se extrae la energia
de emision vertical S1->S0 y la fuerza de oscilador del estado 1 -- mismo calculo que
el chequeo final de 06_s1_optimize_gpu_pcm.py, pero recorrido sobre toda la ensemble
de snapshots de la MD en vez de una sola geometria.

Base: 6-31g* (SIN funciones difusas). Version anterior usaba 6-31+g*; bajada tras el
smoke test en para-nitroanilina (ver README.md, seccion "Decision registrada
sobre la base") -- eleccion deliberada y aceptada, no una demostracion de optimalidad.

Metodologia consistente en los 4 puntos del pipeline (ver README.md, seccion
"Tier 1->3 con PCM + TDDFT"): mismo solvente/base/metodo que 04_s0_optimize_pcm.py,
06_s1_optimize_gpu_pcm.py y tddft_ase_calculator.py (method="TDDFT" en la MD).

NO es una modificacion de 09_ensemble_emission.py -- ese script es el artefacto
historico de Tier 2 (PCM+base difusa+TDDFT sobre el ensemble de WIGNER, ya corrido y
documentado) y no se toca aca. Este es un archivo nuevo para Tier 3 (ensemble de MD),
con su propio --samples/--out por defecto para no pisar los resultados de Tier 2.

Checkpointing: igual que 09_ensemble_emission.py -- se escribe (append) una fila al
CSV apenas se completa cada geometria, y si el CSV de salida ya existe con K filas
completas, se retoma desde la muestra K+1.

Uso:
  python3 13_ensemble_emission_pcm.py --gpu
  (default: CPU, s1_md_samples.xyz, nstates=5, checkpoint-every=5,
   out=ensemble_emission_pcm_tier3.csv)
"""
import argparse
import csv
import os
import time
from pyscf import gto, dft

XC = "wb97x-d3bj"
BASIS = "6-31g*"             # sin funciones difusas -- decision registrada, ver README.md
SOLVENT_METHOD = "IEF-PCM"
SOLVENT_EPS = 24.55          # etanol a 25 C
STATE = 1  # S1: primer estado excitado
# excitation_state/nstates_computed/state_gap_next_eV: agregado a pedido de una
# revision del modelo -- documenta que estado se reporta (S1, indice ascendente de
# energia), cuantos estados calculo TDDFT en total (de los cuales solo se reporta uno),
# y que tan lejos esta el siguiente estado (S2) -- una senal barata de cuanto podria
# moverse el resultado si se reportara un estado distinto, sin tener que re-correr nada.
#
# state{2..5}_energy_eV/state{2..5}_osc: agregado en la revision del 2026-08-26 (Task 1,
# item 3) -- pyscf ya calcula el array completo de oscillator_strength() para los 5
# estados en CADA punto de este pipeline, pero hasta ahora solo se guardaba el indice 0
# (S1). Esto permite chequear si S1 es realmente el estado mas brillante (mayor f) en
# cada geometria muestreada, o si algun estado S2-S5 lo supera -- pregunta que antes de
# este agregado no tenia respuesta en ningun CSV/log existente.
CSV_FIELDS = ["sample_index", "energy_eV", "oscillator_strength", "wavelength_nm",
              "scf_tddft_time_s", "excitation_state", "nstates_computed",
              "state_gap_next_eV",
              "state2_energy_eV", "state2_osc", "state3_energy_eV", "state3_osc",
              "state4_energy_eV", "state4_osc", "state5_energy_eV", "state5_osc"]


def read_xyz_frames(path):
    """Lee un XYZ multi-frame (mismo formato que escriben 08_wigner_sample.py y
    12_sample_trajectory.py): bloques de <natm>\\n<comentario>\\n<natm lineas de
    'simbolo x y z'>."""
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
        return max(0, sum(1 for _ in fh) - 1)  # -1 por la fila de encabezado


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

    # Dispatch por metodo enlazado a mf -- tddft.TDDFT(mf, ...) de nivel superior esta
    # confirmado roto (TypeError) en esta version de pyscf.
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
    # Estados S2-S5 (indices 1-4 de TD): guardar aunque no se reporten como "el" estado,
    # para poder chequear despues cual es realmente el mas brillante -- ver CSV_FIELDS.
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
    print(f"[info] PCM: method={SOLVENT_METHOD}, eps={SOLVENT_EPS} (etanol), "
          f"equilibrium_solvation=True")

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
    parser.add_argument("--samples", default="s1_md_samples.xyz")
    parser.add_argument("--out", default="ensemble_emission_pcm_tier3.csv")
    parser.add_argument("--nstates", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5,
                         help="Cada cuantas geometrias imprimir un resumen de progreso "
                              "(el guardado en CSV es siempre por-fila, no por bloque)")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    main()
