"""
Tier 1 (fix), paso 2 (GPU) -- Optimizacion de geometria S1 con PCM -- C102
Mismo procedimiento que 06_s1_optimize_gpu.py (vacio), agregando PCM (IEF-PCM, etanol,
eps=24.55) + TDDFT completo (no TDA -- ver nota de API mas abajo). Parte de
coumarin102_s0opt_pcm.xyz (salida de 04_s0_optimize_pcm.py), no de la geometria S0 de
vacio, para mantener el solvente consistente en toda la cadena Tier1->Tier3 (ver
README.md).

Base: 6-31g* (SIN funciones difusas). Version anterior usaba 6-31+g*; bajada tras el
smoke test en para-nitroanilina (ver README.md, seccion "Decision registrada
sobre la base") -- eleccion deliberada y aceptada, no una demostracion de optimalidad.

maxsteps=100 para la optimizacion S1: limite de seguridad generoso, NO capado
artificialmente bajo -- en el smoke test de PNA, mucho mas chica y rigida que C102,
la optimizacion S1 convergio sola en 10 pasos con un limite de 30; C102 es mas grande
y flexible asi que puede necesitar mas, pero 100 deja margen amplio. Si llega a tocar
este limite sin converger, es una señal real de que revisar, no un limite a subir
mas en silencio.

TDA -> TDDFT: se cambio de TDA a TDDFT completo para que S0 opt, S1 opt, cada paso de
la MD y el paso de emision por muestra usen todos el mismo nivel de teoria de estado
excitado, no solo el mismo solvente. Confirmado por inspeccion de codigo (NO por
ejecucion directa, a diferencia de la version TDA de este mismo script que si se
corrio como prueba): gpu4pyscf/grad/tdrhf.py::grad_elec bifurca explicitamente en
`is_tda = isinstance(td_grad.base, TDA)`, y gpu4pyscf/solvent/tdscf/pcm.py::
make_tdscf_gradient_object es generico sobre la clase TD que envuelve -- no hay
restriccion a TDA en el codigo de gradiente ni en el wrapper de PCM. Esto NO es lo
mismo que haberlo corrido: la primera ejecucion real de esta combinacion (TDDFT +
PCM + gradiente + GPU) va a ser la primera vez que se corra este script.

Nota de fisica/aproximacion (importante, no ocultar): la solvatacion "equilibrium" que
exige gpu4pyscf para el gradiente TDDFT-PCM (ver mas abajo) relaja el solvente por
completo a la densidad del estado excitado en cada paso de la optimizacion S1, y
tambien se usa para el chequeo final de emision vertical. Esto es una aproximacion
estandar (solvatacion de equilibrio para ambos estados) pero NO es identica a un
tratamiento "no-equilibrio" mas riguroso, en el que solo la polarizacion rapida/
electronica del solvente respondaria instantaneamente durante la transicion vertical
S1->S0, mientras la polarizacion lenta/orientacional queda congelada en su
configuracion equilibrada con S1. El resultado de este script es por lo tanto una
primera aproximacion al efecto de solvente, no el tratamiento mas riguroso posible --
si el numero resultante sigue lejos de literatura, la solvatacion no-equilibrio seria
el siguiente refinamiento a intentar. Esta es una decision ya tomada para este
pipeline, no una opcion a reconsiderar en silencio.

Requisito de la API (confirmado por prueba directa en este entorno, gpu4pyscf 1.7.6,
para la variante TDA -- ver nota "TDA -> TDDFT" arriba sobre por que la variante
TDDFT NO se probo por ejecucion directa, solo por inspeccion de codigo): el gradiente
analitico de PCM-TD en GPU exige `mf.TDDFT(equilibrium_solvation=True)` (dispatch por
metodo enlazado a mf, NO `tddft.TDDFT(mf, equilibrium_solvation=True)` -- esa forma de
nivel superior lanza TypeError en esta version de pyscf, el mismo bug que se encontro
en 02_production_gpu.py/Step 3 del Tier 1). Sin equilibrium_solvation=True,
.Gradients() lanza RuntimeError explicito pidiendo este flag.

Requiere: coumarin102_s0opt_pcm.xyz (correr 04_s0_optimize_pcm.py primero),
pip install geometric, gpu4pyscf-cuda12x, cutensor-cu12.
Uso: python3 06_s1_optimize_gpu_pcm.py | tee s1_optimize_gpu_pcm.log
"""
import time
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

XC = "wb97x-d3bj"
BASIS = "6-31g*"      # sin funciones difusas -- decision registrada, ver README.md
NSTATES = 5
STATE = 1  # S1: primer estado excitado
SOLVENT_EPS = 24.55   # etanol a 25 C, mismo punto de comparacion que 04_s0_optimize_pcm.py
SOLVENT_METHOD = "IEF-PCM"

t0 = time.time()
mol = gto.M(atom="coumarin102_s0opt_pcm.xyz", basis=BASIS, verbose=4)
print(f"[info] {mol.natm} atomos, {mol.nao} funciones de base, basis={BASIS}")
print(f"[info] PCM: method={SOLVENT_METHOD}, eps={SOLVENT_EPS} (etanol), equilibrium_solvation=True")

mf = dft.RKS(mol)
mf.xc = XC
mf = mf.to_gpu()          # <- clave: mueve el objeto SCF a GPU4PySCF
mf = mf.PCM()
mf.with_solvent.method = SOLVENT_METHOD
mf.with_solvent.eps = SOLVENT_EPS

td = mf.TDDFT(equilibrium_solvation=True)
td.nstates = NSTATES

td_grad_scanner = td.Gradients().as_scanner(state=STATE)
mol_s1 = optimize(td_grad_scanner, maxsteps=100)
t_opt = time.time()
print(f"[timing][GPU] Optimizacion geometria S1 (PCM): {t_opt - t0:.1f} s")

mol_s1.tofile("coumarin102_s1opt_gpu_pcm.xyz", format="xyz")
print("[DONE][GPU] Geometria S1 optimizada (PCM) guardada en coumarin102_s1opt_gpu_pcm.xyz")

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
print(f"[timing][GPU] SCF+TDDFT en geometria S1 (chequeo emision vertical, PCM): {t_emis - t1:.1f} s")

e_emission_eV = td_s1geom.e[STATE - 1] * 27.2114
f_emission = td_s1geom.oscillator_strength()[STATE - 1]
lambda_emission_nm = 1239.84 / e_emission_eV
print(f"Energia de emision vertical S1->S0 (Tier-1, GPU, PCM/etanol): {e_emission_eV:.3f} eV")
print(f"Fuerza de oscilador: {f_emission:.4f}")
print(f"Longitud de onda de emision (Tier-1, GPU, PCM/etanol): {lambda_emission_nm:.0f} nm")
print("Referencia experimental: lambda_em ~ 466 nm (etanol, Step 6)")

print(f"\n[DONE][GPU] Tiempo total: {time.time()-t0:.1f} s")
