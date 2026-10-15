"""
Tier 2, paso 3 — Analisis vibracional en el minimo S1 — Cumarina 102 (C102)

Este PySCF (ver README.md para la version instalada) NO tiene Hessiano analitico de
estados excitados: pyscf/hessian/ solo implementa Hessianos de estado fundamental
(RHF/RKS/UHF/UKS); no existe un modulo equivalente para TDA/TDDFT. GPU4PySCF tampoco lo
provee a la fecha de este workflow. Por eso este script implementa un Hessiano NUMERICO
por diferencias finitas centradas del gradiente analitico de TDA (td.Gradients()) en el
estado 1, desplazando cada coordenada cartesiana +/- STEP_ANG (default 0.005 Ang).

Costo: 2 * 3 * natm evaluaciones de gradiente TDA (SCF + TDA + gradiente cada una). Para
C102 (36 atomos) son 216 evaluaciones — esto es lo que hay que presupuestar en GPU antes
de correr esto para moleculas mas grandes.

Una vez construido el Hessiano (natm,natm,3,3) en Ha/Bohr^2, se reusa
pyscf.hessian.thermo.harmonic_analysis (la misma rutina de analisis vibracional de
PySCF para estados fundamentales) para hacer el mass-weighting, proyectar traslacion y
rotacion, diagonalizar, y obtener frecuencias (cm^-1) y modos normales. Esto NO es
especifico de estado excitado: harmonic_analysis solo necesita un array de Hessiano y no
le importa de que metodo salio.

SUPUESTO a verificar (afecta correccion cientifica, no solo de estilo): con un paso de
0.005 Ang se asume que el orden de los 5 estados de TDA (por energia ascendente) no
cambia entre geometrias vecinas, de modo que "el primer estado" (indice STATE-1) sigue
siendo S1 en cada evaluacion. Si hay un cruce evitado (avoided crossing) muy cerca del
minimo S1 esto podria fallar silenciosamente (Hessiano con una fila/columna contaminada
por otro estado). No se implemento tracking de estado por overlap de vectores de
excitacion porque added complexity fuera del alcance pedido; si las frecuencias salen
con muchos modos imaginarios o valores absurdos, esto es lo primero a sospechar.

Uso:
  python3 07_s1_frequencies.py --gpu   # opcional; ver --help para --step-ang, --nstates
  (default: CPU, coumarin102_s1opt.xyz, step=0.005 Ang, nstates=5)

Salida: s1_normal_modes.npz (freq_wavenumber, norm_mode, reduced_mass, mass, atom_symbols,
atom_coords_bohr, hess, step_ang, xc, basis, nstates) para wigner_sample.py (paso 4).

NOTA sobre --gpu: la densidad inicial (dm0) se reusa entre los 216 puntos como numpy
array; si la version instalada de gpu4pyscf no la sube automaticamente a GPU, el primer
error a esperar es un mismatch de tipo/backend en mf.kernel(dm0=...) — en ese caso correr
sin --gpu (el costo de este paso es igual de grande en ambos casos: son 216 SCF+TDA+grad
independientes, GPU solo acelera cada uno).
"""
import argparse
import time
import numpy as np
from pyscf import gto, dft, tddft, lib
from pyscf.hessian import thermo

XC = "wb97x-d3bj"
BASIS = "6-31g*"
STATE = 1  # S1: primer estado excitado


def s1_gradient(coords_bohr, symbols, basis, gpu, dm0=None):
    """SCF (wb97x-d3bj/BASIS) + TDA + gradiente TDA del estado S1 en una geometria dada.
    Devuelve (grad (natm,3) en Ha/Bohr, dm convergida para reusar como init guess)."""
    atom = list(zip(symbols, coords_bohr * lib.param.BOHR))
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = XC
    if gpu:
        mf = mf.to_gpu()
    mf.kernel(dm0=dm0)
    assert mf.converged, "SCF no convergio en un punto de la diferencia finita"

    td = tddft.TDA(mf)
    td.nstates = args.nstates
    td.kernel()

    grad = td.Gradients().kernel(state=STATE)
    dm_out = mf.make_rdm1()
    if gpu:
        dm_out = dm_out.get() if hasattr(dm_out, "get") else np.asarray(dm_out)
    return np.asarray(grad), dm_out


def main():
    t0 = time.time()
    mol0 = gto.M(atom=args.geometry, basis=BASIS, verbose=4)
    natm = mol0.natm
    symbols = [mol0.atom_pure_symbol(i) for i in range(natm)]
    coords0 = mol0.atom_coords()  # Bohr
    step_bohr = args.step_ang / lib.param.BOHR

    print(f"[info] {natm} atomos, basis={BASIS}, xc={XC}, nstates={args.nstates}")
    print(f"[info] paso de diferencia finita: {args.step_ang} Ang = {step_bohr:.6f} Bohr")
    print(f"[info] evaluaciones de gradiente planeadas: {2*3*natm} (2 x 3 x {natm} atomos)")

    # dm0 inicial: SCF en la geometria de referencia (no desplazada), reusado como guess
    # inicial de densidad en cada punto desplazado para acelerar convergencia del SCF.
    mf0 = dft.RKS(mol0)
    mf0.xc = XC
    if args.gpu:
        mf0 = mf0.to_gpu()
    mf0.kernel()
    assert mf0.converged, "SCF de referencia no convergio"
    dm_ref = mf0.make_rdm1()
    if args.gpu:
        dm_ref = dm_ref.get() if hasattr(dm_ref, "get") else np.asarray(dm_ref)

    hess = np.zeros((natm, natm, 3, 3))
    n_evals = 0
    for q in range(natm):
        for y in range(3):
            coords_p = coords0.copy()
            coords_p[q, y] += step_bohr
            t_eval0 = time.time()
            grad_p, _ = s1_gradient(coords_p, symbols, BASIS, args.gpu, dm0=dm_ref)
            n_evals += 1
            print(f"[timing] gradiente {n_evals}/{2*3*natm} (+, atomo {q} dir {y}): "
                  f"{time.time()-t_eval0:.1f} s")

            coords_m = coords0.copy()
            coords_m[q, y] -= step_bohr
            t_eval0 = time.time()
            grad_m, _ = s1_gradient(coords_m, symbols, BASIS, args.gpu, dm0=dm_ref)
            n_evals += 1
            print(f"[timing] gradiente {n_evals}/{2*3*natm} (-, atomo {q} dir {y}): "
                  f"{time.time()-t_eval0:.1f} s")

            hess[:, q, :, y] = (grad_p - grad_m) / (2 * step_bohr)

    # Simetrizar para reducir ruido numerico de la diferencia finita.
    hess = 0.5 * (hess + hess.transpose(1, 0, 3, 2))
    t_hess = time.time()
    print(f"[timing] Hessiano numerico completo: {n_evals} evaluaciones de gradiente, "
          f"{t_hess - t0:.1f} s total")

    results = thermo.harmonic_analysis(mol0, hess, imaginary_freq=False)
    freq_cm = results["freq_wavenumber"]
    norm_mode = results["norm_mode"]         # (nmodes, natm, 3), Bohr por unidad de Q
    reduced_mass = results["reduced_mass"]   # amu

    n_imaginary = int(np.sum(freq_cm < 0))
    print(f"[info] {len(freq_cm)} modos totales, {n_imaginary} con frecuencia imaginaria "
          "(reportada como negativa)")
    if n_imaginary > 0:
        print("[WARN] Modos imaginarios en el 'minimo' S1: la geometria puede no estar "
              "totalmente relajada, o puede tratarse de un punto de silla. Revisar "
              "05/06_s1_optimize antes de confiar en el muestreo de Wigner.")
    print("Frecuencias (cm^-1):", [f"{f:.1f}" for f in freq_cm])

    np.savez(
        args.out,
        freq_wavenumber=freq_cm,
        norm_mode=norm_mode,
        reduced_mass=reduced_mass,
        mass=mol0.atom_mass_list(isotope_avg=True),
        atom_symbols=np.array(symbols),
        atom_coords_bohr=coords0,
        hess=hess,
        step_ang=args.step_ang,
        xc=XC,
        basis=BASIS,
        nstates=args.nstates,
        state=STATE,
    )
    print(f"[DONE] Modos normales guardados en {args.out}")
    print(f"[timing] Tiempo total: {time.time()-t0:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="coumarin102_s1opt.xyz")
    parser.add_argument("--out", default="s1_normal_modes.npz")
    parser.add_argument("--step-ang", type=float, default=0.005,
                         help="Paso de diferencia finita central, en Angstrom (default 0.005)")
    parser.add_argument("--nstates", type=int, default=5)
    parser.add_argument("--gpu", action="store_true",
                         help="Correr cada evaluacion SCF+TDA+gradiente en GPU4PySCF (.to_gpu())")
    args = parser.parse_args()
    main()
