"""
Tier 3, paso 1 -- Dinamica molecular ab initio en la superficie S1 -- Cumarina 102 (C102)

Reemplaza el muestreo estatico de Wigner (Tier 2, pasos 3-4: 07_s1_frequencies.py +
08_wigner_sample.py) por una trayectoria real: parte de la geometria S1 YA OPTIMIZADA
CON PCM en GPU (06_s1_optimize_gpu_pcm.py -> coumarin102_s1opt_gpu_pcm.xyz -- no la
version de vacio coumarin102_s1opt_gpu.xyz, para mantener el solvente consistente
desde el primer frame de la trayectoria), le asigna velocidades termicas de
Maxwell-Boltzmann (removiendo momento lineal Y angular neto, ver mas abajo), y la deja
evolucionar sobre la superficie de energia S1 con PCM
(tddft_ase_calculator.TDAExcitedStateCalculator, method="TDDFT", solvent="IEF-PCM"/
etanol) con un termostato de Langevin (NVT -- ver nota de staging mas abajo).

Metodologia PCM consistente en los 4 puntos del pipeline (ver README.md, seccion
"Tier 1->3 con PCM + TDDFT"): IEF-PCM/etanol/eps=24.55, equilibrium_solvation=True,
base 6-31g* (SIN funciones difusas -- version anterior usaba 6-31+g*, bajada tras el
smoke test en para-nitroanilina; ver README.md, seccion "Decision registrada
sobre la base" -- eleccion deliberada y aceptada, no una demostracion de optimalidad),
TDDFT completo (no TDA) en los 4 puntos -- S0 opt (04_s0_optimize_pcm.py), S1 opt
(06_s1_optimize_gpu_pcm.py), cada paso de esta MD, y el paso de emision por muestra
(13_ensemble_emission_pcm.py).

Staging NVT (confirmado, no hay NPT en ningun lado): no hay caja periodica ni celda
para C102 aislada, asi que no hay presion/volumen que tenga sentido controlar -- solo
Langevin (NVT). La secuencia real es: minimizacion S1 (ya hecha, Tier 1, con PCM) ->
equilibrado NVT (Langevin, primeros n_burnin pasos) -> produccion NVT (Langevin,
resto de los pasos) -> muestreo de snapshots (12_sample_trajectory.py, filtra por
--burnin-steps). Equilibrado y produccion son la MISMA corrida continua de Langevin --
se distinguen solo por la etiqueta de log ("burnin"/"produccion") y por el filtro de
12_sample_trajectory.py, no por dos objetos de dinamica distintos.

Parametros de la corrida (aprobados en el plan de Fase 1 de Tier 3, ver README.md):
  - paso de tiempo: 0.5 fs -- RESTRICCION DURA: el modo mas rapido del Hessiano de
    Tier 2 esta a 3254.3 cm^-1 (periodo ~10.2 fs); un integrador estable necesita
    dt <~ periodo/10-20.
  - termostato: Langevin, T=300 K, friccion=0.02 fs^-1 (tiempo de relajacion 50 fs) --
    SUPOSICION, elegida mas alta (relajacion mas rapida) que un valor "de libro"
    (~100 fs) para poder pagar el equilibrado dentro del presupuesto de tiempo de
    esta maquina; no es un valor derivado de la fisica especifica de C102.
  - equilibrado (burn-in): 100 pasos = 50 fs (reducido de 200, a pedido explicito del
    usuario tras ver el costo estimado del plan original), no se muestrea para el
    espectro. SUPOSICION, guiada por presupuesto -- minimo, no riguroso.
  - produccion: 200 pasos = 100 fs (reducido de 400, misma proporcion, mismo pedido
    explicito). El modo mas LENTO del Hessiano (49.6 cm^-1, periodo ~670 fs) sigue
    siendo mucho mas largo que toda esta trayectoria, ahora mas corta que en el plan
    original -- este script SIGUE NO siendo un reemplazo estadisticamente equivalente
    al muestreo de Wigner de Tier 2 para los modos lentos/flexibles, es un chequeo
    de dinamica real, no una ensemble estadisticamente completa.
  - xc/basis/nstates/state: wb97x-d3bj / 6-31g* / 5 / S1 -- SIN funciones difusas
    (version anterior de este script usaba 6-31+g*; bajada tras el smoke test en
    para-nitroanilina, ver README.md seccion "Decision registrada sobre la
    base" -- eleccion deliberada y aceptada, no una demostracion de optimalidad) y
    TDDFT completo (no TDA), consistente con 04_s0_optimize_pcm.py,
    06_s1_optimize_gpu_pcm.py y 13_ensemble_emission_pcm.py. A diferencia de la
    version original de este script (vacio, TDA, 6-31g*), esta version SI aplica PCM
    en la dinamica misma, no solo en el paso de emision.

ADVERTENCIA (ver tddft_ase_calculator.py, ADVERTENCIA de metodo): TDA+PCM+gradiente+
GPU esta confirmado por ejecucion directa en una sesion anterior de este proyecto.
TDDFT completo+PCM+gradiente+GPU (lo que este script usa) esta soportado por el
codigo fuente inspeccionado de gpu4pyscf (sin restriccion a TDA en el gradiente ni en
el wrapper de PCM) y confirmado funcionando end-to-end en el smoke test v2 (PNA, 10
pasos de optimizacion S1 + 10 pasos de MD, todos con TDDFT+PCM+GPU+gradiente) -- pero
la primera corrida real en C102 (molecula mas grande y flexible que PNA) sigue siendo
la primera vez que esta combinacion exacta se ejecuta a esta escala.

Costo esperado (REVISADO con datos reales del smoke test v2 -- ver README.md,
seccion de costo bajo Tier 3, para el detalle completo de la extrapolacion): el
smoke test v2 midio 130.5 s/paso promedio en PNA (152 funciones de base, 6-31g*,
TDDFT+PCM+GPU, gradiente real de MD). Extrapolado a C102 (300 funciones de base a
6-31g*) por dos metodos independientes (factor de escala empirico PNA->C102 medido
en sesion anterior, y exponente de escala derivado del propio smoke test):
~550 s/paso (promedio de ambos metodos) -> 300 pasos totales * 550 s/paso ~=
165000 s ~= ~45.8 h (~1.9 dias) solo para la MD, mas ~4-6 h de una sola vez (no por
paso) para S0+S1 opt -> estimado total realista ~50-52 h (~2.1-2.2 dias), bajando de
la estimacion original de ~82.5 h.

Checkpointing: cada paso se agrega (append) al archivo de trayectoria ASE
(s1_md_trajectory.traj, formato binario de ase.io.trajectory.Trajectory) y se
sobreescribe un marcador JSON (s1_md_resume.json) con el indice del ultimo paso
completado. Si el proceso se cae o se corta, --resume lee el ultimo frame de la
trayectoria (posiciones + velocidades, que ASE guarda por defecto en cada frame) y
sigue corriendo Langevin desde ahi en vez de arrancar de cero. La matriz de densidad
de warm-start (dm0) NO se checkpointea entre corridas -- perderla en un resume solo
cuesta un SCF inicial mas lento (guess frio), no afecta la correccion del resultado.

ADVERTENCIA: requiere ASE (`pip install ase`), no incluido en el entorno
"fluorescence" al momento de escribir este script -- instalar antes de correr.

Uso:
  python3 11_s1_md_run.py --gpu | tee s1_md_run.log
  python3 11_s1_md_run.py --gpu --resume | tee -a s1_md_run.log   # si se cayo a mitad
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

XC = "wb97x-d3bj"
BASIS = "6-31g*"      # sin funciones difusas -- decision registrada, ver README.md
NSTATES = 5
STATE = 1  # S1: primer estado excitado
METHOD = "TDDFT"       # no TDA -- ver ADVERTENCIA de metodo en tddft_ase_calculator.py
SOLVENT_METHOD = "IEF-PCM"
SOLVENT_EPS = 24.55    # etanol a 25 C, misma convencion que 04/06_..._pcm.py

RESUME_MARKER = "s1_md_resume.json"


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
        Stationary(atoms)     # remueve momento lineal neto del centro de masa
        ZeroRotation(atoms)   # remueve momento angular neto -- sin esto, parte de la
                               # energia termica se va a rotacion rigida de toda la
                               # molecula en vez de a vibracion interna
        print(f"[info] {len(atoms)} atomos, geometria inicial: {args.geometry}")
        print(f"[info] velocidades iniciales Maxwell-Boltzmann a "
              f"{args.temperature_k} K, momento lineal y angular neto removidos")

    print(f"[info] xc={XC}, basis={BASIS}, nstates={NSTATES}, state=S{STATE}, "
          f"method={METHOD}")
    print(f"[info] PCM: method={SOLVENT_METHOD}, eps={SOLVENT_EPS} (etanol), "
          f"equilibrium_solvation=True")
    print(f"[info] dt={args.dt_fs} fs, friccion={args.friction_fs_inv} fs^-1 "
          f"(relajacion {1.0/args.friction_fs_inv:.0f} fs), T={args.temperature_k} K")
    print(f"[info] burn-in={n_burnin} pasos ({n_burnin*args.dt_fs:.0f} fs), "
          f"produccion={n_production} pasos ({n_production*args.dt_fs:.0f} fs)")
    print(f"[info] pasos totales planeados: {n_total} "
          f"(evaluaciones SCF+TDA+gradiente, GPU={'si' if args.gpu else 'no'})")

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
    parser.add_argument("--geometry", default="coumarin102_s1opt_gpu_pcm.xyz")
    parser.add_argument("--traj", default="s1_md_trajectory.traj")
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--friction-fs-inv", type=float, default=0.02)
    parser.add_argument("--burnin-steps", type=int, default=100)
    parser.add_argument("--production-steps", type=int, default=200)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main()
