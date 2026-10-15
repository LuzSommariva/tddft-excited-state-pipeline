"""
Tier 2, paso 2 (CPU) — Optimizacion de geometria S1 (primer estado excitado) — C102
Parte de coumarin102_s0opt.xyz (ver 04_s0_optimize.py). Construye TDA (nstates=5) sobre
wb97x-d3bj/6-31g* y usa el gradiente analitico de TDA (td.Gradients()) para el estado 1
como scanner de geomeTRIC. Version GPU equivalente: 06_s1_optimize_gpu.py (mismos
parametros, mf movido con .to_gpu()) — correr ambas y comparar geometria/energia final
es el control de correccion numerica para este paso, igual que 01 vs 02 en Tier 1.

Al final, corre un SCF+TDA de punto simple en la geometria S1 optimizada: esta es la
emision vertical S1->S0 "Tier-1" (single point), usada como chequeo de cordura inmediato
contra el maximo experimental ~466-470 nm (etanol/agua), antes de invertir tiempo en el
analisis de frecuencias (Tier 2, paso 3).

NOTA de convencion: state=1 en pyscf (td.Gradients().as_scanner(state=1)) es el PRIMER
estado excitado (S1), 1-indexado — no el estado 0/base.

Requiere: coumarin102_s0opt.xyz (generado por 04_s0_optimize.py), pip install geometric.
Uso: python3 05_s1_optimize_cpu.py | tee s1_optimize_cpu.log
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

td = tddft.TDA(mf)
td.nstates = NSTATES

# El scanner corre SCF+TDA+gradiente TDA en cada paso del optimizador; no hace falta
# mf.kernel()/td.kernel() explicitos antes (mismo patron que examples/geomopt/12-excited_states.py).
td_grad_scanner = td.Gradients().as_scanner(state=STATE)
mol_s1 = optimize(td_grad_scanner, maxsteps=100)
t_opt = time.time()
print(f"[timing] Optimizacion geometria S1 (CPU): {t_opt - t0:.1f} s")

mol_s1.tofile("coumarin102_s1opt.xyz", format="xyz")
print("[DONE] Geometria S1 optimizada guardada en coumarin102_s1opt.xyz")

# --- Chequeo Tier-1: emision vertical S1->S0 en la geometria S1 optimizada ---
t1 = time.time()
mf_s1geom = dft.RKS(mol_s1)
mf_s1geom.xc = XC
mf_s1geom.kernel()
assert mf_s1geom.converged, "SCF no convergio en geometria S1 optimizada"

td_s1geom = tddft.TDA(mf_s1geom)
td_s1geom.nstates = NSTATES
td_s1geom.kernel()
t_emis = time.time()
print(f"[timing] SCF+TDA en geometria S1 (chequeo emision vertical): {t_emis - t1:.1f} s")

e_emission_eV = td_s1geom.e[STATE - 1] * 27.2114
f_emission = td_s1geom.oscillator_strength()[STATE - 1]
lambda_emission_nm = 1239.84 / e_emission_eV
print(f"Energia de emision vertical S1->S0 (Tier-1): {e_emission_eV:.3f} eV")
print(f"Fuerza de oscilador: {f_emission:.4f}")
print(f"Longitud de onda de emision (Tier-1): {lambda_emission_nm:.0f} nm")
print("Referencia experimental: lambda_em ~ 466-470 nm (etanol/agua)")

print(f"\n[DONE] Tiempo total: {time.time()-t0:.1f} s")
