"""
Fase 0 — Setup y sanity check de GPU4PySCF.
SCF trivial de agua (RKS/wb97x-d/6-31g*) en CPU y despues en GPU con .to_gpu().
Mientras corre el bloque GPU, mirar `nvidia-smi` en otra terminal (o `nvidia-smi dmon`)
para confirmar actividad real (GPU-Util > 0%, memoria usada).

Requiere:
  pip install pyscf
  pip install gpu4pyscf-cuda12x     # o gpu4pyscf-cuda11x segun tu CUDA (nvidia-smi / nvcc --version)
  pip install cutensor-cu12         # (cutensor-cu11 si corresponde)

Uso: python3 00_sanity_check.py
"""
import time
from pyscf import gto, dft

# pyscf>=2.14 bloquea "wb97x-d" (ambiguo entre D3/D4); wb97x-d3bj (wb97x-v + D3BJ) es el
# equivalente moderno soportado. Ver pyscf/scf/dispersion.py _black_list/_white_list.
XC = "wb97x-d3bj"
BASIS = "6-31g*"

water = """
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
"""

mol = gto.M(atom=water, basis=BASIS, verbose=3)

print("=== CPU ===")
t0 = time.time()
mf_cpu = dft.RKS(mol)
mf_cpu.xc = XC
mf_cpu.kernel()
assert mf_cpu.converged, "SCF CPU no convergio"
print(f"[CPU] E_tot = {mf_cpu.e_tot:.8f} Ha | {time.time()-t0:.2f} s")

print("\n=== GPU (GPU4PySCF) ===")
print("[info] Mira 'nvidia-smi' ahora para confirmar actividad en la GPU...")
t0 = time.time()
mf_gpu = dft.RKS(mol)
mf_gpu.xc = XC
mf_gpu = mf_gpu.to_gpu()
mf_gpu.kernel()
assert mf_gpu.converged, "SCF GPU no convergio"
print(f"[GPU] E_tot = {mf_gpu.e_tot:.8f} Ha | {time.time()-t0:.2f} s")

de = abs(mf_cpu.e_tot - mf_gpu.e_tot)
print(f"\nDiferencia CPU vs GPU: {de:.2e} Ha  {'[OK, <1e-5]' if de < 1e-5 else '[REVISAR, >1e-5]'}")
print("[DONE] Sanity check completo.")
