"""
Tier 1 (fix), paso 2 (GPU) -- Optimizacion de geometria S1 con PCM -- p-terphenyl

Mismo procedimiento que 06_s1_optimize_gpu_pcm.py (Cumarina 102): PCM + TDDFT
completo (no TDA), mf.TDDFT(equilibrium_solvation=True) via dispatch enlazado a mf
(la forma de nivel superior tddft.TDDFT(mf,...) esta rota en esta version de
pyscf). Parametrizado via molecule_config.py -- ver ese script para el detalle de
las decisiones de API/fisica que este pipeline ya tomo y no reconsidera en
silencio (equilibrium_solvation, TDA->TDDFT, maxsteps=100 sin capar).

Parte de p_terphenyl_s0opt_pcm.xyz (salida de 04_s0_optimize_pcm_pterphenyl.py).
NO toca ningun archivo de Cumarina 102.

Requiere: p_terphenyl_s0opt_pcm.xyz (correr 04_s0_optimize_pcm_pterphenyl.py primero),
pip install geometric, gpu4pyscf-cuda12x, cutensor-cu12.
Uso: python3 06_s1_optimize_gpu_pcm_pterphenyl.py | tee s1_optimize_gpu_pcm_pterphenyl.log
"""
import time
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

from molecule_config import XC, BASIS, MOLECULES

cfg = MOLECULES["p_terphenyl"]
NSTATES = 5
STATE = 1  # S1: primer estado excitado
SOLVENT_EPS = cfg["solvent_eps"]
SOLVENT_METHOD = cfg["solvent_method"]

t0 = time.time()
mol = gto.M(atom="p_terphenyl_s0opt_pcm.xyz", basis=BASIS, verbose=4)
print(f"[info] {mol.natm} atomos, {mol.nao} funciones de base, basis={BASIS}")
print(f"[info] PCM: method={SOLVENT_METHOD}, eps={SOLVENT_EPS} ({cfg['solvent_name']}), "
      f"equilibrium_solvation=True")

mf = dft.RKS(mol)
mf.xc = XC
mf = mf.to_gpu()
mf = mf.PCM()
mf.with_solvent.method = SOLVENT_METHOD
mf.with_solvent.eps = SOLVENT_EPS

td = mf.TDDFT(equilibrium_solvation=True)
td.nstates = NSTATES

td_grad_scanner = td.Gradients().as_scanner(state=STATE)
mol_s1 = optimize(td_grad_scanner, maxsteps=100)
t_opt = time.time()
print(f"[timing][GPU] Optimizacion geometria S1 (PCM): {t_opt - t0:.1f} s")

mol_s1.tofile("p_terphenyl_s1opt_gpu_pcm.xyz", format="xyz")
print("[DONE][GPU] Geometria S1 optimizada (PCM) guardada en p_terphenyl_s1opt_gpu_pcm.xyz")

# --- Chequeo Tier-1: emision vertical S1->S0 en la geometria S1 optimizada (GPU, PCM) ---
t1 = time.time()
mf_s1geom = dft.RKS(mol_s1)
mf_s1geom.xc = XC
mf_s1geom = mf_s1geom.to_gpu()
mf_s1geom = mf_s1geom.PCM()
mf_s1geom.with_solvent.method = SOLVENT_METHOD
mf_s1geom.with_solvent.eps = SOLVENT_EPS
mf_s1geom.kernel()
assert mf_s1geom.converged, "SCF no convergio en geometria S1 optimizada (GPU, PCM)"

td_s1geom = mf_s1geom.TDDFT(equilibrium_solvation=True)
td_s1geom.nstates = NSTATES
td_s1geom.kernel()
t_emis = time.time()
print(f"[timing][GPU] SCF+TDDFT en geometria S1 (chequeo emision vertical, PCM): "
      f"{t_emis - t1:.1f} s")

e_emission_eV = td_s1geom.e[STATE - 1] * 27.2114
f_emission = td_s1geom.oscillator_strength()[STATE - 1]
lambda_emission_nm = 1239.84 / e_emission_eV
print(f"Energia de emision vertical S1->S0 (Tier-1, GPU, PCM/{cfg['solvent_name']}): "
      f"{e_emission_eV:.3f} eV")
print(f"Fuerza de oscilador: {f_emission:.4f}")
print(f"Longitud de onda de emision (Tier-1, GPU, PCM/{cfg['solvent_name']}): "
      f"{lambda_emission_nm:.0f} nm")
print(f"Referencia experimental: lambda_em ~ {cfg['exp_emission_nm']:.0f} nm "
      f"({cfg['exp_reference']})")

print(f"\n[DONE][GPU] Tiempo total: {time.time()-t0:.1f} s")
