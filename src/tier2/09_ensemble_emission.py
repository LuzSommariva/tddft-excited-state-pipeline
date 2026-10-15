"""
Tier 2, paso 5 — Emision vertical sobre la ensemble de Wigner — Cumarina 102 (C102)

Para cada geometria muestreada por 08_wigner_sample.py: SCF + TDDFT, y se extrae la
energia de emision vertical S1->S0 y la fuerza de oscilador del estado 1 (asi es como el
Nuclear Ensemble Approach construye una banda en vez de un pico unico: la dispersion de
geometrias termicas/de punto-cero alrededor del minimo S1 se traduce en una dispersion de
energias verticales).

MODIFICACION (Tier 1, Option A — ver sesion del 2026-08-13/14): el minimo S1 y su
Hessiano (07_s1_frequencies.py) NO se recalcularon con solvente/base mejor -- seguimos
usando el unico minimo S1 real que tenemos, el de vacio (TDA/6-31g*, coumarin102_s1opt_gpu.xyz).
Lo que SI se aplica aqui, por geometria de la ensemble, es la correccion validada en los
diagnosticos de un solo punto (_diag_pcm_singlepoint.py, _diag_pcm_diffuse_basis.py,
_diag_pcm_tddft.py): PCM implicito (IEF-PCM, etanol, eps=24.55) + base con funciones
difusas (6-31+g*) + TDDFT completo en vez de TDA (que fue lo que cerro el 42% del gap a
literatura en esos diagnosticos, vacio -> PCM). Esto es una
correccion de un solo punto sobre geometrias vacio-optimizadas, NO una re-optimizacion
completa con estos parametros (esa es la Option B, mas cara y no ejecutada esta corrida).

API: `mf.TDDFT(equilibrium_solvation=True)` (dispatch por metodo enlazado a mf) -- NO
`tddft.TDDFT(mf, ...)`, que falla con TypeError en esta version de pyscf/gpu4pyscf (ver
Step 3 del Tier 1, mismo bug). equilibrium_solvation=True se pasa aunque aqui no se pide
gradiente, porque asi se valido el numero en _diag_pcm_tddft.py -- afecta la energia de
excitacion en si (cuanto se relaja el solvente a la densidad excitada), no solo el
gradiente.

Checkpointing: se escribe (append) una fila al CSV cada vez que se completa una
geometria, no se acumula todo en memoria y se vuelca al final — si el proceso se cae o
se corta por timeout, se pierde a lo sumo el trabajo de la geometria en curso. Ademas, si
el CSV de salida ya existe con K filas completas, el script retoma desde la muestra K+1
en vez de volver a empezar (mismo criterio que --checkpoint-every, aunque el nombre del
flag se refiere solo a la frecuencia de flush/print, no a la unidad de resumen: se
retoma por fila, y se hace flush a disco en cada fila para que un corte a mitad de un
bloque de N no pierda mas de una geometria).

Uso:
  python3 09_ensemble_emission.py --gpu
  (default: CPU, s1_wigner_samples.xyz, nstates=5, checkpoint-every=5, out=ensemble_emission.csv)
"""
import argparse
import csv
import os
import time
from pyscf import gto, dft

XC = "wb97x-d3bj"
BASIS = "6-31+g*"            # Tier 1 Option A: base difusa, validada en _diag_pcm_diffuse_basis.py
SOLVENT_METHOD = "IEF-PCM"   # Tier 1 Option A: PCM implicito, validado en _diag_pcm_singlepoint.py
SOLVENT_EPS = 24.55          # etanol a 25 C
STATE = 1  # S1: primer estado excitado
# excitation_state/nstates_computed/state_gap_next_eV: agregado a pedido de una
# revision del modelo, aplicado aqui solo para consistencia de esquema hacia adelante
# -- el CSV historico de esta corrida (ensemble_emission.csv, N=40, ya completo) NO se
# retoma/recalcula con este cambio; ver 13_ensemble_emission_pcm.py para el razonamiento
# completo del agregado.
CSV_FIELDS = ["sample_index", "energy_eV", "oscillator_strength", "wavelength_nm",
              "scf_tddft_time_s", "excitation_state", "nstates_computed",
              "state_gap_next_eV"]


def read_xyz_frames(path):
    """Lee un XYZ multi-frame (formato escrito por 08_wigner_sample.py):
    bloques de <natm>\\n<comentario>\\n<natm lineas de 'simbolo x y z'>."""
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

    td = mf.TDDFT(equilibrium_solvation=True)
    td.nstates = args.nstates
    td.kernel()

    e_eV = float(td.e[STATE - 1]) * 27.2114
    f_osc = float(td.oscillator_strength()[STATE - 1])
    if STATE < args.nstates:
        gap_next_eV = float((td.e[STATE] - td.e[STATE - 1]) * 27.2114)
    else:
        gap_next_eV = float("nan")
    return e_eV, f_osc, gap_next_eV


def main():
    frames = read_xyz_frames(args.samples)
    n_total = len(frames)
    print(f"[info] {n_total} geometrias cargadas de {args.samples}")

    n_done = n_completed_rows(args.out)
    if n_done > 0:
        print(f"[info] {args.out} ya tiene {n_done} filas completas; retomando desde la "
              f"muestra {n_done}")
    else:
        with open(args.out, "w", newline="") as fh:
            csv.writer(fh).writerow(CSV_FIELDS)

    t0 = time.time()
    for idx in range(n_done, n_total):
        _, atoms = frames[idx]
        t_start = time.time()
        e_eV, f_osc, gap_next_eV = emission_at(atoms, args.gpu)
        dt = time.time() - t_start
        lambda_nm = 1239.84 / e_eV

        with open(args.out, "a", newline="") as fh:
            csv.writer(fh).writerow([idx, f"{e_eV:.6f}", f"{f_osc:.6f}", f"{lambda_nm:.2f}",
                                      f"{dt:.1f}", STATE, args.nstates,
                                      f"{gap_next_eV:.6f}"])

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
    parser.add_argument("--samples", default="s1_wigner_samples.xyz")
    parser.add_argument("--out", default="ensemble_emission.csv")
    parser.add_argument("--nstates", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5,
                         help="Cada cuantas geometrias imprimir un resumen de progreso "
                              "(el guardado en CSV es siempre por-fila, no por bloque)")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    main()
