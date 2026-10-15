"""
Tier 2, paso 1 — Optimizacion de geometria S0 (estado fundamental) — Cumarina 102 (C102)
Parte de coumarin102.xyz (geometria MMFF, sin optimizar con DFT) y la optimiza con
wb97x-d3bj/6-31g* usando el optimizador geomeTRIC (pyscf.geomopt.geometric_solver).
La geometria resultante alimenta 05_s1_optimize_cpu.py / 06_s1_optimize_gpu.py.

Requiere:
  pip install geometric   # libreria geomeTRIC usada por pyscf.geomopt.geometric_solver

Uso: python3 04_s0_optimize.py | tee s0_optimize.log
"""
import time
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

XC = "wb97x-d3bj"
BASIS = "6-31g*"

t0 = time.time()
mol = gto.M(atom="coumarin102.xyz", basis=BASIS, verbose=4)
print(f"[info] {mol.natm} atomos, {mol.nao} funciones de base, basis={BASIS}")

mf = dft.RKS(mol)
mf.xc = XC

# optimize() corre su propio ciclo SCF en cada paso; no hace falta mf.kernel() antes.
mol_eq = optimize(mf, maxsteps=100)
t_opt = time.time()
print(f"[timing] Optimizacion geometria S0: {t_opt - t0:.1f} s")

# SCF final en la geometria optimizada, para reportar una energia convergida limpia.
mf_eq = dft.RKS(mol_eq)
mf_eq.xc = XC
mf_eq.kernel()
assert mf_eq.converged, "SCF final en geometria S0 optimizada no convergio"
print(f"[info] E_tot(S0, optimizado) = {mf_eq.e_tot:.6f} Ha")

mol_eq.tofile("coumarin102_s0opt.xyz", format="xyz")
print("[DONE] Geometria S0 optimizada guardada en coumarin102_s0opt.xyz")
print(f"[timing] Tiempo total: {time.time()-t0:.1f} s")
