"""
Cálculo de PRODUCCION (GPU, GPU4PySCF) — Cumarina 102 (C102)
Mismos parametros que 01_production_cpu_reference.py, para comparar
correccion numerica y medir speed-up real GPU vs CPU.

Requiere:
  pip install pyscf
  pip install gpu4pyscf-cuda12x     # o gpu4pyscf-cuda11x segun tu CUDA (chequear con nvidia-smi / nvcc --version)
  pip install cutensor-cu12         # (cutensor-cu11 si corresponde)

Uso: python3 02_production_gpu.py | tee gpu_run.log
"""
import time
from pyscf import gto, dft, tddft

# pyscf>=2.14 bloquea "wb97x-d" (ambiguo entre D3/D4); wb97x-d3bj (wb97x-v + D3BJ) es el
# equivalente moderno soportado. Ver pyscf/scf/dispersion.py _black_list/_white_list.
XC = "wb97x-d3bj"
BASIS = "6-31g*"
NSTATES = 5

t0 = time.time()
mol = gto.M(atom="coumarin102.xyz", basis=BASIS, verbose=4)
print(f"[info] {mol.natm} atomos, {mol.nao} funciones de base, basis={BASIS}")

mf = dft.RKS(mol)
mf.xc = XC
mf = mf.to_gpu()          # <- clave: mueve el objeto SCF a GPU4PySCF
mf.kernel()
assert mf.converged, "SCF no convergio en GPU"
t_scf = time.time()
print(f"[timing][GPU] SCF: {t_scf - t0:.1f} s | E_tot = {mf.e_tot:.6f} Ha")

# --- TDA en GPU ---
t1 = time.time()
tda = tddft.TDA(mf)
tda.nstates = NSTATES
tda.kernel()
t_tda = time.time()
print(f"[timing][GPU] TDA ({NSTATES} estados): {t_tda - t1:.1f} s")
print("Energias TDA (eV):", [f"{e*27.2114:.3f}" for e in tda.e])

# --- TDDFT completo en GPU ---
t2 = time.time()
td = tddft.TDDFT(mf)
td.nstates = NSTATES
td.kernel()
t_tddft = time.time()
print(f"[timing][GPU] TDDFT completo ({NSTATES} estados): {t_tddft - t2:.1f} s")
print("Energias TDDFT (eV):", [f"{e*27.2114:.3f}" for e in td.e])
print("Fuerzas de oscilador TDDFT:", [f"{o:.4f}" for o in td.oscillator_strength()])

print(f"\n[DONE][GPU] Tiempo total: {time.time()-t0:.1f} s")

# --- Chequeo rapido: to_cpu() para confirmar interoperabilidad (Fase 4 del plan) ---
mf_cpu_check = mf.to_cpu()
print(f"[check] to_cpu() interoperability OK, E_tot recomputable = {mf_cpu_check.e_tot:.6f} Ha")
