"""
Tier 1 (fix), paso 1 -- Optimizacion de geometria S0 con solvente implicito (PCM) -- C102
Mismo procedimiento que 04_s0_optimize.py (vacio), agregando PCM (IEF-PCM, etanol,
eps=24.55) para probar cuanto del corrimiento al azul observado en
vacio (ver 06_s1_optimize_gpu.py + comparacion con literatura) se explica por la falta
de solvente explicito.

Base: 6-31g* (SIN funciones difusas). Version anterior de este script usaba 6-31+g*;
bajada tras el smoke test en para-nitroanilina (ver README.md, seccion
"Decision registrada sobre la base") -- eleccion deliberada y aceptada para esta
corrida (fisicamente razonable para esta transicion de valencia, y necesaria para
que la corrida completa entre en presupuesto de tiempo), NO una demostracion de que
sea la base optima.

Metodologia consistente en los 4 puntos del pipeline Tier1->Tier3 con PCM para esta
molecula (ver README.md, seccion "Tier 1->3 con PCM + TDDFT"): este paso (S0),
06_s1_optimize_gpu_pcm.py (S1), tddft_ase_calculator.py (MD) y
13_ensemble_emission_pcm.py (emision por muestra) usan TODOS el mismo solvente
(IEF-PCM/etanol/eps=24.55), la misma base (6-31g*) y, donde aplica estado excitado,
TDDFT completo (no TDA).

No sobreescribe 04_s0_optimize.py ni su salida (coumarin102_s0opt.xyz): este script
parte del mismo coumarin102.xyz de entrada y escribe coumarin102_s0opt_pcm.xyz, para
poder comparar vacio vs PCM directamente.

PCM en pyscf/gpu4pyscf es una feature experimental (advertencia propia del paquete al
usarla) -- API puede cambiar en versiones futuras.

Requiere: pip install geometric (ya instalado en el env fluorescence).
Uso: python3 04_s0_optimize_pcm.py | tee s0_optimize_pcm.log
"""
import time
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

XC = "wb97x-d3bj"
BASIS = "6-31g*"      # sin funciones difusas -- decision registrada, ver README.md
SOLVENT_EPS = 24.55   # etanol a 25 C; punto de comparacion de Step 6 (lambda_em lit. ~466 nm)
SOLVENT_METHOD = "IEF-PCM"

t0 = time.time()
mol = gto.M(atom="coumarin102.xyz", basis=BASIS, verbose=4)
print(f"[info] {mol.natm} atomos, {mol.nao} funciones de base, basis={BASIS}")
print(f"[info] PCM: method={SOLVENT_METHOD}, eps={SOLVENT_EPS} (etanol)")

mf = dft.RKS(mol)
mf.xc = XC
mf = mf.PCM()
mf.with_solvent.method = SOLVENT_METHOD
mf.with_solvent.eps = SOLVENT_EPS

# optimize() corre su propio ciclo SCF en cada paso; no hace falta mf.kernel() antes.
mol_eq = optimize(mf, maxsteps=100)
t_opt = time.time()
print(f"[timing] Optimizacion geometria S0 (PCM): {t_opt - t0:.1f} s")

# SCF final en la geometria optimizada, para reportar una energia convergida limpia.
mf_eq = dft.RKS(mol_eq)
mf_eq.xc = XC
mf_eq = mf_eq.PCM()
mf_eq.with_solvent.method = SOLVENT_METHOD
mf_eq.with_solvent.eps = SOLVENT_EPS
mf_eq.kernel()
assert mf_eq.converged, "SCF final en geometria S0 optimizada (PCM) no convergio"
print(f"[info] E_tot(S0, optimizado, PCM) = {mf_eq.e_tot:.6f} Ha")

mol_eq.tofile("coumarin102_s0opt_pcm.xyz", format="xyz")
print("[DONE] Geometria S0 optimizada (PCM) guardada en coumarin102_s0opt_pcm.xyz")
print(f"[timing] Tiempo total: {time.time()-t0:.1f} s")
