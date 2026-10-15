"""
Calculator ASE generico para energia/fuerzas de un estado excitado TDA/TDDFT
(PySCF/GPU4PySCF), con o sin solvente implicito PCM.

No es especifico de Cumarina 102: cualquier ase.Atoms + basis/xc/nstates/state validos
para PySCF sirven. Pensado para reutilizarse mas adelante en el pipeline de proteinas
(Hybrid_Quantum_mechanical_spectra_generator), no solo en este repo. Por eso method,
solvent_method y solvent_eps son parametros explicitos (default: TDA, sin solvente) en
vez de estar hardcodeados -- este modulo no asume la receta particular de ningun
proyecto que lo use.

Que calcula en cada .calculate(): SCF (RKS) + TDA o TDDFT completo (segun `method`) +
gradiente analitico del estado excitado `state` (default 1 = S1, primer estado
excitado), opcionalmente bajo PCM (solvatacion de equilibrio, requerida por gpu4pyscf
para gradientes de PCM-TD -- ver ADVERTENCIA de solvatacion mas abajo), expuesta como
ase.calculators.calculator.Calculator para que las herramientas de dinamica de ASE
(ase.md.langevin.Langevin, etc.) puedan usarlo como motor de fuerzas.

Uso con PCM/TDDFT (11_s1_md_run.py, run de C102 con solvente): method="TDDFT",
solvent_method="IEF-PCM", solvent_eps=24.55 -- ver README.md, seccion "Tier 1->3
con PCM + TDDFT" para por que estos 3 valores tienen que coincidir con
04_s0_optimize_pcm.py / 06_s1_optimize_gpu_pcm.py / 13_ensemble_emission_pcm.py.

Convenciones de unidades (frontera ASE <-> PySCF), escritas explicitamente:
- Posiciones: ASE usa Angstrom; es tambien la unidad default de pyscf.gto.M(atom=...),
  asi que NO se convierte nada al construir la molecula.
- Energia: PySCF devuelve Hartree (mf.e_tot + td.e[state-1]); ASE espera eV.
    HARTREE2EV = 27.211386245988   (pyscf.data.nist.HARTREE2EV)
- Fuerzas: PySCF devuelve el GRADIENTE (dE/dR) en Hartree/Bohr; ASE espera FUERZA
  (-dE/dR) en eV/Angstrom.
    BOHR = 0.52917721067           (Angstrom por Bohr, pyscf.lib.param.BOHR)
    FORCE_CONV = HARTREE2EV / BOHR ~= 51.422   (eV/Angstrom por Ha/Bohr)
  forces = -grad_ha_bohr * FORCE_CONV   (signo invertido: fuerza = -gradiente)
  Ambas constantes se toman de pyscf en tiempo de import, no se hardcodean por
  separado, para que no puedan desincronizarse de lo que el resto del repo usa.

Warm start: la matriz de densidad convergida de la ultima llamada exitosa se guarda en
self._dm_last y se pasa como dm0= en la siguiente llamada (mismo patron que dm_ref en
07_s1_frequencies.py) -- ayuda a que el SCF converja mas rapido cuando la geometria
cambia poco entre pasos consecutivos de una trayectoria. Se puede desactivar con
warm_start=False si se sospecha que esta contaminando la convergencia.

ADVERTENCIA (mismo supuesto que 07_s1_frequencies.py, pero mas expuesto aca): se asume
que el estado `state` (por indice de energia ascendente) sigue siendo el mismo estado
electronico de un paso al siguiente. En una trayectoria de MD esto es mas fragil que en
las diferencias finitas de 07 (desplazamientos de solo 0.005 Ang): el sistema explora
mucha mas geometria y un cruce evitado entre S1 y S2 es mas probable. No hay tracking
de estado por overlap de vectores de excitacion implementado -- self.results incluye
'state_gap_eV' (diferencia de energia entre el estado `state` y el siguiente por
arriba) para que el script que llama pueda loguearlo/chequearlo y detectar tramos
sospechosos de la trayectoria.

No hace nada especial con GPU mas alla de lo que ya hacen 00-10: gpu=True solo agrega
mf = mf.to_gpu() antes de mf.kernel(), igual que en todo el resto del repo.

ADVERTENCIA de solvatacion (si solvent_eps is not None): se usa `equilibrium_solvation
=True` siempre -- esta es la eleccion fisicamente correcta para todo lo que este
modulo calcula (minimos relajados, snapshots termicos de una MD en equilibrio con el
termostato -- nunca una transicion vertical instantanea "en el acto"), no un default
barato a reconsiderar. gpu4pyscf ademas lo EXIGE para poder pedir `.Gradients()` bajo
PCM (si no, `.Gradients()` lanza RuntimeError explicito).

ADVERTENCIA de metodo (method="TDA" vs "TDDFT"): con PCM, `.Gradients()` sobre TDA
esta confirmado por ejecucion directa en este entorno (gpu4pyscf 1.7.6, sesion previa
de este proyecto). `.Gradients()` sobre TDDFT completo bajo PCM esta soportado por el
codigo (gpu4pyscf/grad/tdrhf.py::grad_elec bifurca explicitamente para TDA vs TDDFT
completo; gpu4pyscf/solvent/tdscf/pcm.py::make_tdscf_gradient_object es generico sobre
la clase TD que envuelve, no hay restriccion a TDA) pero NO se probo por ejecucion
directa antes de usarse en este modulo -- si method="TDDFT" falla con PCM+gradiente en
la practica, es una discrepancia real entre lo que el codigo sugiere y lo que gpu4pyscf
realmente ejecuta, no algo que este modulo deba silenciar o revertir solo -- avisar.
`tddft.TDDFT(mf)` de nivel superior (no bound-method) esta confirmado ROTO en esta
version de pyscf (TypeError) -- por eso el dispatch siempre es `mf.TDDFT(...)`, nunca
`tddft.TDDFT(mf, ...)`.

Requiere: PySCF, gpu4pyscf (si gpu=True), ASE (`pip install ase` -- no incluido en el
entorno "fluorescence" al momento de escribir este modulo).
"""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from pyscf import gto, dft
from pyscf.data import nist
from pyscf import lib

HARTREE2EV = nist.HARTREE2EV     # 27.211386245988
BOHR = lib.param.BOHR            # Angstrom por Bohr, ~0.52917721067
FORCE_CONV = HARTREE2EV / BOHR   # eV/Angstrom por Ha/Bohr, ~51.422


class TDAExcitedStateCalculator(Calculator):
    """Energia y fuerzas de un estado excitado TDA o TDDFT (PySCF/GPU4PySCF), para
    ase.Atoms, con PCM implicito opcional.

    Parametros
    ----------
    xc, basis, nstates : igual convencion que el resto del repo (default
        wb97x-d3bj / 6-31g* / 5).
    state : int, default 1
        Indice del estado excitado (1 = S1, el primer estado excitado por energia).
    method : {"TDA", "TDDFT"}, default "TDA"
        TDA (aproximacion de Tamm-Dancoff) o TDDFT completo (respuesta lineal no
        Hermitiana). Ver ADVERTENCIA de metodo en el docstring del modulo sobre el
        estado de verificacion de cada uno bajo PCM+gradiente+GPU.
    gpu : bool, default False
        Si True, mf = mf.to_gpu() antes de correr el SCF (requiere gpu4pyscf).
    solvent_method : str o None, default None
        Metodo de PCM (p.ej. "IEF-PCM"). None = vacio (sin solvente).
    solvent_eps : float o None, default None
        Constante dielectrica del solvente. None = vacio. Si no es None, se activa
        PCM con equilibrium_solvation=True (ver ADVERTENCIA de solvatacion arriba) --
        no hay manera de pedir PCM sin solvatacion de equilibrio en este modulo, a
        proposito.
    warm_start : bool, default True
        Si True, reusa la densidad convergida del paso anterior como dm0 del
        siguiente SCF (mismo patron que 07_s1_frequencies.py).
    pyscf_verbose : int, default 0
        Verbosity de pyscf para el objeto mol/mf (0 = silencioso; subir a 4 para
        depurar un paso puntual, no recomendado para una trayectoria completa por
        el volumen de log que genera).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, xc="wb97x-d3bj", basis="6-31g*", nstates=5, state=1,
                 method="TDA", gpu=False, solvent_method=None, solvent_eps=None,
                 warm_start=True, pyscf_verbose=0, **kwargs):
        super().__init__(**kwargs)
        assert method in ("TDA", "TDDFT"), f"method invalido: {method!r}"
        self.xc = xc
        self.basis = basis
        self.nstates = nstates
        self.state = state
        self.method = method
        self.gpu = gpu
        self.solvent_method = solvent_method
        self.solvent_eps = solvent_eps
        self.warm_start = warm_start
        self.pyscf_verbose = pyscf_verbose
        self._dm_last = None
        self.n_calls = 0

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        symbols = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()  # Angstrom, sin convertir
        mol = gto.M(atom=list(zip(symbols, positions)), basis=self.basis,
                    verbose=self.pyscf_verbose)

        mf = dft.RKS(mol)
        mf.xc = self.xc
        if self.gpu:
            mf = mf.to_gpu()
        use_solvent = self.solvent_eps is not None
        if use_solvent:
            mf = mf.PCM()
            mf.with_solvent.method = self.solvent_method
            mf.with_solvent.eps = self.solvent_eps

        dm0 = self._dm_last if self.warm_start else None
        mf.kernel(dm0=dm0)
        assert mf.converged, "SCF no convergio en un paso de la trayectoria"

        # Dispatch por metodo enlazado a mf (mf.TDA/mf.TDDFT), nunca por el
        # dispatcher de nivel superior tddft.TDA(mf)/tddft.TDDFT(mf, ...): esta
        # ultima forma esta confirmada rota para TDDFT en esta version de pyscf
        # (TypeError), y equilibrium_solvation solo es aceptado por el dispatch
        # enlazado a mf en ambos casos.
        if use_solvent:
            if self.method == "TDA":
                td = mf.TDA(equilibrium_solvation=True)
            else:
                td = mf.TDDFT(equilibrium_solvation=True)
        else:
            if self.method == "TDA":
                td = mf.TDA()
            else:
                td = mf.TDDFT()
        td.nstates = self.nstates
        td.kernel()

        grad_ha_bohr = np.asarray(td.Gradients().kernel(state=self.state))

        e_ha = mf.e_tot + td.e[self.state - 1]
        self.results["energy"] = e_ha * HARTREE2EV
        self.results["forces"] = -grad_ha_bohr * FORCE_CONV

        # Chequeo de cruce de estados: gap de energia al siguiente estado por arriba.
        # No es un tracking real por overlap -- solo una senal de alarma barata para
        # que el script que llama decida si loguear un [WARN] (ver 11_s1_md_run.py).
        if self.state < self.nstates:
            self.results["state_gap_eV"] = float(
                (td.e[self.state] - td.e[self.state - 1]) * HARTREE2EV
            )
        else:
            self.results["state_gap_eV"] = float("nan")

        dm_out = mf.make_rdm1()
        if self.gpu:
            dm_out = dm_out.get() if hasattr(dm_out, "get") else np.asarray(dm_out)
        self._dm_last = dm_out
        self.n_calls += 1
