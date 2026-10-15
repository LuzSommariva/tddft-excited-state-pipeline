"""
Tier 2, paso 6 — Espectro de emision con ensanchamiento Gaussiano — Cumarina 102 (C102)

Carga el CSV de (energia, fuerza de oscilador) por geometria de 09_ensemble_emission.py
y construye el "stick spectrum" ensanchado:

    I(E) = sum_i  w_i * exp( -(E - E_i)^2 / (2 * sigma^2) )

con sigma en eV (default 0.15 eV — valor tipico usado en NEA para C102-like cumarinas;
ajustable con --sigma-ev, este numero domina el ancho de banda final, no la fisica del
calculo, asi que conviene poder variarlo sin volver a correr la ensemble).

Peso por geometria (--weight-e2f, default apagado = usar solo f_i):
  - f_i sola: heuristica mas comun, proporcional a la seccion eficaz de absorcion en esa
    geometria; es lo que la mayoria de los papers de NEA grafican por defecto.
  - E_i^2 * f_i (con --weight-e2f): mas cercano a la tasa de emision espontanea real. El
    coeficiente de Einstein A (tasa de emision) escala como omega^3 * |mu|^2; como la
    fuerza de oscilador f ya escala como omega * |mu|^2, se tiene que
    A ~ omega^2 * f ~ E^2 * f. Este peso favorece mas a las geometrias de mayor energia
    de emision (mayor E) que el peso por f sola, y en general da un maximo ligeramente
    distinto — es la opcion "fisicamente mas correcta" para una banda de EMISION
    (distinto del caso de absorcion, donde f sola ya es proporcional a la tasa de
    absorcion correcta).

Uso:
  python3 10_broaden_and_plot.py
  python3 10_broaden_and_plot.py --weight-e2f --sigma-ev 0.10
  (ver --help para --csv, --out, --npoints)
"""
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_EM_MIN_NM = 466.0
EXP_EM_MAX_NM = 470.0


def load_csv(path):
    energies, oscs = [], []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            energies.append(float(row["energy_eV"]))
            oscs.append(float(row["oscillator_strength"]))
    return np.array(energies), np.array(oscs)


def main():
    energies_eV, oscs = load_csv(args.csv)
    n = len(energies_eV)
    print(f"[info] {n} geometrias cargadas de {args.csv}")
    if n == 0:
        raise SystemExit(f"[ERROR] {args.csv} no tiene filas — correr 09_ensemble_emission.py primero")

    if args.weight_e2f:
        weights = (energies_eV ** 2) * oscs
        weight_label = "E^2 * f (aprox. tasa de emision)"
    else:
        weights = oscs
        weight_label = "f (fuerza de oscilador)"

    e_grid_eV = np.linspace(energies_eV.min() - 5 * args.sigma_ev,
                             energies_eV.max() + 5 * args.sigma_ev,
                             args.npoints)
    intensity = np.zeros_like(e_grid_eV)
    for e_i, w_i in zip(energies_eV, weights):
        intensity += w_i * np.exp(-((e_grid_eV - e_i) ** 2) / (2 * args.sigma_ev ** 2))

    nm_grid = 1239.84 / e_grid_eV
    peak_idx = int(np.argmax(intensity))
    peak_nm = nm_grid[peak_idx]

    print(f"[info] peso usado: {weight_label}")
    print(f"[info] sigma = {args.sigma_ev} eV")
    print(f"Longitud de onda de emision del pico (ensanchado): {peak_nm:.1f} nm")
    print(f"Referencia experimental: {EXP_EM_MIN_NM:.0f}-{EXP_EM_MAX_NM:.0f} nm (etanol/agua)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nm_grid, intensity, color="tab:blue", lw=1.5,
            label=f"Espectro NEA (ensanchado, sigma={args.sigma_ev} eV)")
    ax.axvspan(EXP_EM_MIN_NM, EXP_EM_MAX_NM, color="tab:orange", alpha=0.25,
               label=f"Experimental ({EXP_EM_MIN_NM:.0f}-{EXP_EM_MAX_NM:.0f} nm)")
    ax.axvline(peak_nm, color="tab:blue", ls="--", lw=1,
               label=f"Pico calculado ({peak_nm:.1f} nm)")
    ax.set_xlabel("Longitud de onda (nm)")
    ax.set_ylabel("Intensidad (u. arb.)")
    ax.set_title(f"Cumarina 102 — espectro de emision NEA (N={n}, peso={weight_label})")
    ax.legend(fontsize=8)
    ax.set_xlim(nm_grid.min(), nm_grid.max())
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[DONE] Espectro guardado en {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="ensemble_emission.csv")
    parser.add_argument("--out", default="emission_spectrum.png")
    parser.add_argument("--sigma-ev", type=float, default=0.15,
                         help="Ancho (desviacion estandar) del ensanchamiento Gaussiano, en eV")
    parser.add_argument("--npoints", type=int, default=2000)
    parser.add_argument("--weight-e2f", action="store_true",
                         help="Pesar cada geometria por E_i^2*f_i en vez de f_i sola "
                              "(mas cercano a la tasa de emision real; ver docstring)")
    args = parser.parse_args()
    main()
