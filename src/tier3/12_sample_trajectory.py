"""
Tier 3, paso 2 -- Muestreo de snapshots de la trayectoria de MD -- Cumarina 102 (C102)

Lee la trayectoria escrita por 11_s1_md_run.py (s1_md_trajectory.traj, formato ASE) y
extrae snapshots cada --interval-steps pasos, empezando despues de --burnin-steps (los
mismos valores usados al lanzar la MD -- se pasan de nuevo aca porque no se leen del
propio archivo de trayectoria).

Escribe el MISMO formato XYZ multi-frame que 08_wigner_sample.py (bloques de
"<natm>\\n<comentario>\\n<natm lineas de 'simbolo x y z'>"), para que
09_ensemble_emission.py pueda consumirlo SIN modificar -- ese script no distingue si
las geometrias vinieron de un muestreo de Wigner (Tier 2) o de una trayectoria de MD
(Tier 3), solo lee bloques XYZ.

Uso:
  python3 12_sample_trajectory.py --burnin-steps 200 --interval-steps 20
  (ver --help para --traj, --out)
"""
import argparse

from ase.io.trajectory import Trajectory


def main():
    traj = Trajectory(args.traj, mode="r")
    n_frames = len(traj)
    print(f"[info] {n_frames} frames en {args.traj}")

    if args.burnin_steps >= n_frames:
        traj.close()
        raise SystemExit(
            f"[ERROR] burnin-steps ({args.burnin_steps}) >= frames disponibles "
            f"({n_frames}) -- la trayectoria no llego a terminar el burn-in, o los "
            "parametros no coinciden con los usados en 11_s1_md_run.py"
        )

    indices = list(range(args.burnin_steps, n_frames, args.interval_steps))
    print(f"[info] muestreando cada {args.interval_steps} pasos despues del paso "
          f"{args.burnin_steps} (burn-in): {len(indices)} snapshots")

    with open(args.out, "w") as fh:
        for isample, idx in enumerate(indices):
            atoms = traj[idx]
            natm = len(atoms)
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()  # Angstrom

            fh.write(f"{natm}\n")
            fh.write(f"MD sample {isample} (frame {idx} de {args.traj}, "
                     f"burnin={args.burnin_steps}, interval={args.interval_steps})\n")
            for symb, (x, y, z) in zip(symbols, positions):
                fh.write(f"{symb:<4s} {x:17.8f} {y:17.8f} {z:17.8f}\n")

    traj.close()
    print(f"[DONE] {len(indices)} geometrias guardadas en {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", default="s1_md_trajectory.traj")
    parser.add_argument("--out", default="s1_md_samples.xyz")
    parser.add_argument("--burnin-steps", type=int, default=200)
    parser.add_argument("--interval-steps", type=int, default=20)
    args = parser.parse_args()
    main()
