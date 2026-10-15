"""
Cálculo de PRODUCCION (CPU, referencia) — Cumarina 102 (C102)
Usa los mismos defaults que qmmm.py del repo: xc='wb97x-d', basis='6-31g*', nstates=5.
Sirve como control de corrección numérica contra la versión GPU (02_production_gpu.py).

Requiere: coumarin102.xyz en el mismo directorio (generado con gen_geometry.py).
Uso: python3 01_production_cpu_reference.py | tee cpu_reference.log
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
mf.kernel()
assert mf.converged, "SCF no convergio"
t_scf = time.time()
print(f"[timing] SCF: {t_scf - t0:.1f} s | E_tot = {mf.e_tot:.6f} Ha")

# --- TDA ---
t1 = time.time()
tda = tddft.TDA(mf)
tda.nstates = NSTATES
tda.kernel()
t_tda = time.time()
print(f"[timing] TDA ({NSTATES} estados): {t_tda - t1:.1f} s")
print("Energias TDA (eV):", [f"{e*27.2114:.3f}" for e in tda.e])
print("Fuerzas de oscilador TDA:", [f"{o:.4f}" for o in tda.oscillator_strength()])

# --- TDDFT completo ---
t2 = time.time()
td = tddft.TDDFT(mf)
td.nstates = NSTATES
td.kernel()
t_tddft = time.time()
print(f"[timing] TDDFT completo ({NSTATES} estados): {t_tddft - t2:.1f} s")
print("Energias TDDFT (eV):", [f"{e*27.2114:.3f}" for e in td.e])
print("Fuerzas de oscilador TDDFT:", [f"{o:.4f}" for o in td.oscillator_strength()])

print(f"\n[DONE] Tiempo total: {time.time()-t0:.1f} s")
print("Referencia experimental: lambda_abs ~ 390 nm (~3.18 eV) en etanol")
print(f"Estado 1 TDDFT -> {1239.84/(td.e[0]*27.2114):.0f} nm (comparar con 390 nm)")
