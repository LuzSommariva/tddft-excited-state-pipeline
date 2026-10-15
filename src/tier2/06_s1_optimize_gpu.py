"""
Tier 2, paso 2 (GPU) — Optimizacion de geometria S1 (primer estado excitado) — C102
Misma receta que 05_s1_optimize_cpu.py, con mf movido a GPU4PySCF via .to_gpu().
Correr este script y 05_s1_optimize_cpu.py y comparar coumarin102_s1opt.xyz / la energia
de emision Tier-1 impresa al final es el control de correccion numerica CPU vs GPU para
este paso (igual espiritu que 03_compare_cpu_gpu.py para Tier 1; no hay un 03 automatico
para este par porque las salidas son geometrias, no solo numeros — comparar a mano).

ADVERTENCIA (verificar antes de correr): el soporte de gradiente analitico TDA/TDDFT en
GPU4PySCF depende de la version instalada. Este script asume que gpu4pyscf provee
`Gradients()` para TDA de forma analoga a pyscf (mismo patron que 02_production_gpu.py
asume que TDA/TDDFT "solo funcionan" tras .to_gpu()). Si `td.Gradients()` no esta
disponible o lanza NotImplementedError en GPU, ese es un limite real de la version
instalada, no un bug de este script — avisar en vez de forzar un workaround silencioso.
En ese caso, la alternativa razonable es optimizar S1 solo en CPU (05) y reservar la GPU
para el paso mas caro de este workflow: el barrido de la ensemble (09_ensemble_emission.py).

Requiere: coumarin102_s0opt.xyz, pip install geometric, gpu4pyscf-cuda12x, cutensor-cu12.
Uso: python3 06_s1_optimize_gpu.py | tee s1_optimize_gpu.log
"""
import time
from pyscf import gto, dft, tddft
from pyscf.geomopt.geometric_solver import optimize

XC = "wb97x-d3bj"
BASIS = "6-31g*"
NSTATES = 5
STATE = 1  # S1: primer estado excitado

t0 = time.time()
mol = gto.M(atom="coumarin102_s0opt.xyz", basis=BASIS, verbose=4)
print(f"[info] {mol.natm} atomos, {mol.nao} funciones de base, basis={BASIS}")

mf = dft.RKS(mol)
mf.xc = XC
mf = mf.to_gpu()          # <- clave: mueve el objeto SCF a GPU4PySCF

td = tddft.TDA(mf)
td.nstates = NSTATES

td_grad_scanner = td.Gradients().as_scanner(state=STATE)
mol_s1 = optimize(td_grad_scanner, maxsteps=100)
t_opt = time.time()
print(f"[timing][GPU] Optimizacion geometria S1: {t_opt - t0:.1f} s")

mol_s1.tofile("coumarin102_s1opt_gpu.xyz", format="xyz")
print("[DONE][GPU] Geometria S1 optimizada guardada en coumarin102_s1opt_gpu.xyz")
print("[info] Comparar a mano contra coumarin102_s1opt.xyz (CPU, 05_s1_optimize_cpu.py).")

# --- Chequeo Tier-1: emision vertical S1->S0 en la geometria S1 optimizada (GPU) ---
t1 = time.time()
mf_s1geom = dft.RKS(mol_s1)
mf_s1geom.xc = XC
mf_s1geom = mf_s1geom.to_gpu()
mf_s1geom.kernel()
assert mf_s1geom.converged, "SCF no convergio en geometria S1 optimizada (GPU)"

td_s1geom = tddft.TDA(mf_s1geom)
td_s1geom.nstates = NSTATES
td_s1geom.kernel()
t_emis = time.time()
print(f"[timing][GPU] SCF+TDA en geometria S1 (chequeo emision vertical): {t_emis - t1:.1f} s")

e_emission_eV = td_s1geom.e[STATE - 1] * 27.2114
f_emission = td_s1geom.oscillator_strength()[STATE - 1]
lambda_emission_nm = 1239.84 / e_emission_eV
print(f"Energia de emision vertical S1->S0 (Tier-1, GPU): {e_emission_eV:.3f} eV")
print(f"Fuerza de oscilador: {f_emission:.4f}")
print(f"Longitud de onda de emision (Tier-1, GPU): {lambda_emission_nm:.0f} nm")
print("Referencia experimental: lambda_em ~ 466-470 nm (etanol/agua)")

print(f"\n[DONE][GPU] Tiempo total: {time.time()-t0:.1f} s")
