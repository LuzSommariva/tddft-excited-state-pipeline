"""
Tier 2, paso 4 — Muestreo de Wigner alrededor del minimo S1 — Cumarina 102 (C102)

Carga los modos normales de 07_s1_frequencies.py (s1_normal_modes.npz) y genera N
geometrias desplazadas siguiendo la distribucion de Wigner del oscilador armonico en su
estado vibracional fundamental (v=0), una por cada modo normal incluido.

Formula (posicion unicamente; no hace falta muestrear momentos porque solo vamos a
calcular energias verticales, no dinamica): para el modo k con frecuencia angular
omega_k [rad/s] y masa reducida mu_k [kg], la distribucion de Wigner del punto cero en
la coordenada normal Q_k [m] es gaussiana con:
    sigma_k = sqrt(hbar / (2 * mu_k * omega_k))
El desplazamiento cartesiano real es Delta_x = sum_k Q_k * norm_mode[k], con Q_k ~
Normal(0, sigma_k) (convertido a Bohr, que es la unidad de norm_mode segun lo guarda
07_s1_frequencies.py). Esta es la formula estandar de muestreo de Wigner posicional usada
en el Nuclear Ensemble Approach (ver p.ej. Crespo-Otero & Barbatti, Theor. Chem. Acc. 131,
1237 (2012), y los scripts wigner.py de SHARC/Newton-X).

Modos excluidos del muestreo: frecuencias imaginarias (guardadas como negativas por
07_s1_frequencies.py) y frecuencias por debajo de --min-freq-cm (default 10 cm^-1, para
evitar dividir por ~0 en modos residuales de traslacion/rotacion mal proyectados
numericamente). Esas coordenadas normales simplemente no se desplazan (se dejan en 0).

Uso:
  python3 08_wigner_sample.py --n 100 --seed 42
  (ver --help para --modes, --min-freq-cm, --out)

Salida: s1_wigner_samples.xyz, multi-frame XYZ (N bloques de natm+2 lineas, uno por
muestra) para que 09_ensemble_emission.py lo consuma y para poder inspeccionar a mano
unas pocas muestras en cualquier visor de xyz (VMD, Avogadro, etc.).
"""
import argparse
import numpy as np
from pyscf import lib
from pyscf.data import nist


def main():
    data = np.load(args.modes)
    freq_cm = data["freq_wavenumber"]
    norm_mode = data["norm_mode"]              # (nmodes, natm, 3), Bohr por unidad de Q
    reduced_mass_amu = data["reduced_mass"]     # amu
    symbols = data["atom_symbols"]
    coords0_bohr = data["atom_coords_bohr"]     # (natm, 3)
    natm = coords0_bohr.shape[0]
    nmodes = len(freq_cm)

    included = freq_cm > args.min_freq_cm
    n_excluded = nmodes - int(np.sum(included))
    print(f"[info] {nmodes} modos normales cargados de {args.modes}")
    print(f"[info] {n_excluded} modos excluidos del muestreo (imaginarios o "
          f"< {args.min_freq_cm} cm^-1)")
    print(f"[info] {int(np.sum(included))} modos incluidos en el muestreo de Wigner")

    # sigma_k [Bohr] para cada modo incluido
    omega = 2 * np.pi * nist.LIGHT_SPEED_SI * (freq_cm * 1e2)      # rad/s (freq_cm en cm^-1 -> m^-1 *1e2)
    mu_kg = reduced_mass_amu * nist.ATOMIC_MASS                     # kg
    sigma_m = np.zeros(nmodes)
    sigma_m[included] = np.sqrt(nist.HBAR / (2 * mu_kg[included] * omega[included]))
    sigma_bohr = sigma_m / nist.BOHR_SI

    rng = np.random.default_rng(args.seed)

    with open(args.out, "w") as fh:
        for isample in range(args.n):
            Q = np.zeros(nmodes)
            Q[included] = rng.normal(0.0, sigma_bohr[included])
            disp_bohr = np.einsum("k,kri->ri", Q, norm_mode)
            coords_bohr = coords0_bohr + disp_bohr
            coords_ang = coords_bohr * lib.param.BOHR

            fh.write(f"{natm}\n")
            fh.write(f"Wigner sample {isample} (seed={args.seed}, "
                     f"n_modes_included={int(np.sum(included))})\n")
            for a in range(natm):
                x, y, z = coords_ang[a]
                fh.write(f"{symbols[a]:<4s} {x:17.8f} {y:17.8f} {z:17.8f}\n")

    print(f"[DONE] {args.n} geometrias de Wigner guardadas en {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="Numero de muestras (default 100)")
    parser.add_argument("--modes", default="s1_normal_modes.npz")
    parser.add_argument("--out", default="s1_wigner_samples.xyz")
    parser.add_argument("--min-freq-cm", type=float, default=10.0,
                         help="Frecuencia minima (cm^-1) para incluir un modo en el muestreo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main()
