"""
Tier 1 (fix), paso 1 -- Optimizacion de geometria S0 con solvente implicito (PCM) --
p-terphenyl

Mismo procedimiento que 04_s0_optimize_pcm.py (Cumarina 102), parametrizado via
molecule_config.py en vez de un segundo archivo con las constantes de C102
copiadas y cambiadas a mano. Solvente/base/xc para p-terphenyl (cyclohexano,
eps=2.02) segun lo confirmado en la Fase 3 (README no actualizado todavia --
ver el reporte de esa fase en la conversacion).

NO toca ningun archivo de Cumarina 102 (coumarin102.xyz, coumarin102_s0opt_pcm.xyz,
04_s0_optimize_pcm.py): entrada/salida son p_terphenyl.xyz / p_terphenyl_s0opt_pcm.xyz,
nombres de archivo distintos a proposito.

Requiere: p_terphenyl.xyz (correr gen_geometry_pterphenyl.py primero),
pip install geometric (ya instalado en el env fluorescence).
Uso: python3 04_s0_optimize_pcm_pterphenyl.py | tee s0_optimize_pcm_pterphenyl.log
"""
import time
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

from molecule_config import XC, BASIS, MOLECULES

cfg = MOLECULES["p_terphenyl"]
SOLVENT_EPS = cfg["solvent_eps"]
SOLVENT_METHOD = cfg["solvent_method"]

t0 = time.time()
mol = gto.M(atom="p_terphenyl.xyz", basis=BASIS, verbose=4)
print(f"[info] {mol.natm} atomos, {mol.nao} funciones de base, basis={BASIS}")
print(f"[info] PCM: method={SOLVENT_METHOD}, eps={SOLVENT_EPS} ({cfg['solvent_name']})")

mf = dft.RKS(mol)
mf.xc = XC
mf = mf.PCM()
mf.with_solvent.method = SOLVENT_METHOD
mf.with_solvent.eps = SOLVENT_EPS

mol_eq = optimize(mf, maxsteps=100)
t_opt = time.time()
print(f"[timing] Optimizacion geometria S0 (PCM): {t_opt - t0:.1f} s")

mf_eq = dft.RKS(mol_eq)
mf_eq.xc = XC
mf_eq = mf_eq.PCM()
mf_eq.with_solvent.method = SOLVENT_METHOD
mf_eq.with_solvent.eps = SOLVENT_EPS
mf_eq.kernel()
assert mf_eq.converged, "SCF final en geometria S0 optimizada (PCM) no convergio"
print(f"[info] E_tot(S0, optimizado, PCM) = {mf_eq.e_tot:.6f} Ha")

mol_eq.tofile("p_terphenyl_s0opt_pcm.xyz", format="xyz")
print("[DONE] Geometria S0 optimizada (PCM) guardada en p_terphenyl_s0opt_pcm.xyz")
print(f"[timing] Tiempo total: {time.time()-t0:.1f} s")
