"""
Tier 3, paso 1 -- Dinamica molecular ab initio en la superficie S1 -- p-terphenyl

Mismo procedimiento que 11_s1_md_run.py (Cumarina 102), parametrizado via
molecule_config.py. Parametros de la corrida IDENTICOS a los ya validados en la
corrida real de C102 (Fase 3, confirmado explicitamente: no adoptar ninguna de las
mejoras propuestas en el analisis de esa corrida) -- dt=0.5 fs, friccion=0.02 fs^-1,
T=300 K, burn-in=100 pasos, produccion=200 pasos, nstates=5, TDDFT completo (no TDA).

IMPORTANTE -- archivos de salida con nombres DISTINTOS a los de Cumarina 102 a
proposito (s1_md_trajectory_pterphenyl.traj, s1_md_resume_pterphenyl.json): si este
script usara los mismos nombres que 11_s1_md_run.py, el primer paso de esta MD
sobreescribiria s1_md_resume.json (que documenta el estado final ya completo de la
corrida real de C102) y potencialmente confundiria una corrida con la otra. El
marcador de resume NO es un argumento de linea de comandos en el script original
(esta hardcodeado), asi que la unica forma segura de reusar la logica es un archivo
separado con su propio nombre de marcador hardcodeado, no pasar --traj distinto y
asumir que alcanza.

Requiere: ASE (`pip install ase`, ya instalado en el env fluorescence para C102).
Uso:
  python3 11_s1_md_run_pterphenyl.py --gpu | tee s1_md_run_pterphenyl.log
  python3 11_s1_md_run_pterphenyl.py --gpu --resume | tee -a s1_md_run_pterphenyl.log
"""
import argparse
import json
import os
import time

from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, Stationary,
                                          ZeroRotation)

from tddft_ase_calculator import TDAExcitedStateCalculator
from molecule_config import XC, BASIS, MOLECULES

cfg = MOLECULES["p_terphenyl"]
NSTATES = 5
STATE = 1  # S1: primer estado excitado
METHOD = "TDDFT"
SOLVENT_METHOD = cfg["solvent_method"]
SOLVENT_EPS = cfg["solvent_eps"]

RESUME_MARKER = "s1_md_resume_pterphenyl.json"  # distinto del de C102 a proposito


def make_calculator(gpu):
    return TDAExcitedStateCalculator(xc=XC, basis=BASIS, nstates=NSTATES, state=STATE,
                                      method=METHOD, gpu=gpu,
                                      solvent_method=SOLVENT_METHOD,
                                      solvent_eps=SOLVENT_EPS, warm_start=True)


def load_resume_state(traj_path):
    if not (os.path.exists(RESUME_MARKER) and os.path.exists(traj_path)):
        return None
    with open(RESUME_MARKER) as fh:
        marker = json.load(fh)
    traj = Trajectory(traj_path, mode="r")
    if len(traj) == 0:
        traj.close()
        return None
    last_atoms = traj[-1]
    traj.close()
    return marker, last_atoms


def main():
    t0 = time.time()
    dt = args.dt_fs * units.fs
    friction = args.friction_fs_inv / units.fs
    n_burnin = args.burnin_steps
    n_production = args.production_steps
    n_total = n_burnin + n_production

    calc = make_calculator(args.gpu)

    resumed = False
    step_start = 0
    if args.resume:
        state = load_resume_state(args.traj)
        if state is not None:
            marker, atoms = state
            atoms.calc = calc
            step_start = marker["last_step"] + 1
            resumed = True
            print(f"[info] Resumiendo desde el paso {step_start}/{n_total} "
                  f"(marcador: {RESUME_MARKER})")
        else:
            print("[WARN] --resume pedido pero no hay trayectoria/marcador previos "
                  "utilizables; arrancando de cero")

    if not resumed:
        atoms = read(args.geometry)
        atoms.calc = calc
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_k)
        Stationary(atoms)
        ZeroRotation(atoms)
        print(f"[info] {len(atoms)} atomos, geometria inicial: {args.geometry}")
        print(f"[info] velocidades iniciales Maxwell-Boltzmann a "
              f"{args.temperature_k} K, momento lineal y angular neto removidos")

    print(f"[info] xc={XC}, basis={BASIS}, nstates={NSTATES}, state=S{STATE}, "
          f"method={METHOD}")
    print(f"[info] PCM: method={SOLVENT_METHOD}, eps={SOLVENT_EPS} "
          f"({cfg['solvent_name']}), equilibrium_solvation=True")
    print(f"[info] dt={args.dt_fs} fs, friccion={args.friction_fs_inv} fs^-1 "
          f"(relajacion {1.0/args.friction_fs_inv:.0f} fs), T={args.temperature_k} K")
    print(f"[info] burn-in={n_burnin} pasos ({n_burnin*args.dt_fs:.0f} fs), "
          f"produccion={n_production} pasos ({n_production*args.dt_fs:.0f} fs)")
    print(f"[info] pasos totales planeados: {n_total} "
          f"(evaluaciones SCF+TDDFT+gradiente, GPU={'si' if args.gpu else 'no'})")

    dyn = Langevin(atoms, dt, temperature_K=args.temperature_k, friction=friction)

    traj_mode = "a" if resumed else "w"
    traj = Trajectory(args.traj, mode=traj_mode, atoms=atoms)

    state_box = {"count": 0, "t_last": time.time()}

    def on_step():
        abs_step = step_start + state_box["count"]
        dt_wall = time.time() - state_box["t_last"]
        state_box["t_last"] = time.time()

        e_pot = atoms.get_potential_energy()
        t_inst = atoms.get_temperature()
        gap = atoms.calc.results.get("state_gap_eV", float("nan"))

        phase = "burnin" if abs_step < n_burnin else "produccion"
        print(f"[timing] paso {abs_step+1}/{n_total} ({phase}): "
              f"E_pot={e_pot:.4f} eV, T_inst={t_inst:.1f} K, "
              f"gap_S{STATE}->S{STATE+1}={gap:.3f} eV | {dt_wall:.1f} s")

        if gap < 0.05:
            print(f"[WARN] gap entre S{STATE} y S{STATE+1} muy chico "
                  f"({gap:.3f} eV) en el paso {abs_step+1} -- posible cruce de "
                  "estados, revisar esta zona de la trayectoria antes de confiar "
                  "en ella para el espectro")

        traj.write(atoms)
        with open(RESUME_MARKER, "w") as fh:
            json.dump({"last_step": abs_step, "n_total": n_total,
                       "dt_fs": args.dt_fs, "temperature_k": args.temperature_k,
                       "friction_fs_inv": args.friction_fs_inv}, fh)

        state_box["count"] += 1

    dyn.attach(on_step, interval=1)

    steps_remaining = n_total - step_start
    if steps_remaining <= 0:
        print(f"[info] Ya se completaron {step_start} de {n_total} pasos segun "
              f"{RESUME_MARKER}; nada que hacer.")
    else:
        dyn.run(steps_remaining)

    traj.close()
    print(f"[DONE] Trayectoria completa: {n_total} pasos en {args.traj}")
    print(f"[timing] Tiempo total (esta corrida): {time.time()-t0:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="p_terphenyl_s1opt_gpu_pcm.xyz")
    parser.add_argument("--traj", default="s1_md_trajectory_pterphenyl.traj")
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--friction-fs-inv", type=float, default=0.02)
    parser.add_argument("--burnin-steps", type=int, default=100)
    parser.add_argument("--production-steps", type=int, default=200)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main()
