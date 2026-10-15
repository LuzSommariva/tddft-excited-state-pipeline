# Theory Behind a Tier 3 (Excited-State AIMD) Emission Spectrum

This is a conceptual explanation only — no implementation, no code, no assumptions about
any particular pipeline. It's organized around the physical justification for each stage
of the calculation, independent of how it's coded.

---

## 1. Why optimize the ground-state geometry first, and why it's the starting point for everything

Every step here rests on the **Born–Oppenheimer approximation**: because nuclei are
thousands of times heavier than electrons, the electrons relax to their instantaneous
ground state almost instantly compared to how fast the nuclei move. This lets you split
the problem in two: for any fixed set of nuclear positions, solve for the electronic
energy; then treat that electronic energy, as a function of nuclear positions, as a
**potential energy surface (PES)** that the nuclei move on. There's one PES for the ground
electronic state (S0) and a separate one for each excited state (S1, S2, ...) — they're
different surfaces because a different electron distribution pulls on the nuclei
differently.

Optimizing the ground-state geometry means finding the minimum of the S0 PES: the
arrangement of atoms where every force (the gradient of the energy) is zero. Physically,
this is the geometry a molecule actually sits in at rest, in the dark, before anything
excites it — a real, physically meaningful starting configuration, not an arbitrary guess.
It's the starting point for everything downstream because **absorption starts from
whatever geometry the molecule was already sitting in** — you can't compute "what happens
when light hits the molecule" without first knowing what the molecule looked like the
instant before the light arrived.

---

## 2. The Franck–Condon point and why nuclei "don't move" during absorption

The **Franck–Condon principle** says that electronic transitions (absorption or emission)
happen much faster than nuclei can move — roughly 10⁻¹⁵ s for the electronic jump versus
10⁻¹³–10⁻¹² s for a single bond vibration. So during the instant of absorption, the nuclei
are effectively frozen in whatever geometry they had right before the photon arrived — for
absorption, that's the **S0 equilibrium geometry** from Step 1.

The **Franck–Condon point** is exactly that: the point on the *excited-state* PES directly
above the ground-state minimum, evaluated at the ground-state geometry but with the
electron distribution promoted to the excited state. This is called a **vertical
excitation** because on an energy-vs-geometry diagram, the transition is drawn as a
straight vertical line — same x-coordinate (geometry), jumping to a higher y-coordinate
(energy) on a different curve. It's not the excited state's true minimum — it's just
"the excited-state energy evaluated at the geometry you started from," because that's
the only geometry available in the instant the transition happens.

**Analogy:** think of a ball sitting at the bottom of one valley (S0 minimum). If you
suddenly and instantaneously teleport it onto a *different* landscape (the S1 PES) without
letting it move, it lands wherever that new landscape happens to be directly above where it
was — not necessarily anywhere near *that* landscape's own lowest point.

---

## 3. Why emission needs a relaxed excited-state geometry — and what the Stokes shift is

Absorption and emission are not mirror images of the same event, because of what happens
*between* them. Once a molecule is excited (say, by absorbing light at the Franck–Condon
point), it does *not* emit immediately from there. Instead:

1. The molecule now sits on the S1 PES, but not at S1's minimum — it landed at whatever
   point was vertically above the S0 geometry, which is generally *not* a low-energy point
   on the S1 surface.
2. On the excited-state PES, that point has non-zero forces on it, so the nuclei relax —
   bond lengths and angles shift — sliding downhill on S1 until they reach the **S1
   equilibrium geometry** (the excited-state minimum). This relaxation is extremely fast
   (picoseconds or less) compared to how long the molecule stays excited before it emits
   (nanoseconds for a typical fluorophore) — so by the time emission happens, essentially
   every molecule doing the emitting has already reached the S1 minimum. This is the
   physical basis of **Kasha's rule**: emission overwhelmingly occurs from the lowest
   vibrational level of the lowest excited state, regardless of which state/level
   absorption originally populated.
3. Emission is then itself a *vertical* transition too, but now starting from the **S1
   minimum**, dropping straight down to whatever the S0 PES looks like at that (excited
   state's) geometry — which is generally *not* the S0 minimum either, since the geometry
   has changed while the molecule was excited.

Because absorption happens from the S0 minimum (evaluated on the S1 surface) and emission
happens from the S1 minimum (evaluated on the S0 surface), the two transition energies are
different — emission is lower in energy (longer wavelength) than absorption. This
gap is the **Stokes shift**, and it's a direct, measurable signature of how much the
geometry (bond lengths, ring planarity, etc.) actually changes between the ground and
excited electronic states. A molecule that barely changes shape upon excitation (rigid,
planar dyes) has a small Stokes shift; a molecule that distorts a lot has a large one.

This is exactly *why* emission calculations can't skip the excited-state geometry
optimization the way a bare absorption calculation sometimes can get away with a single
vertical point: emission genuinely happens from a *different, relaxed* geometry, not the
ground-state one.

---

## 4. What excited-state molecular dynamics adds beyond just optimizing S1's minimum

Optimizing the S1 minimum gives you one number: the emission energy at absolute zero
temperature, from a single, perfectly relaxed geometry. Real molecules, at any real
temperature, are never sitting exactly at that minimum — they're constantly vibrating
around it, with a spread of instantaneous geometries determined by their vibrational
motion and thermal energy (k_BT). Running molecular dynamics *on the excited-state
surface* lets the nuclei explore that spread of geometries the way a real, warm molecule
actually would, instead of freezing them at the single lowest-energy point.

This matters for the emission spectrum for two related reasons:

- **Vibronic coupling / vibrational sampling:** the emission energy is not a single fixed
  number — it depends on exactly which vibrational configuration the molecule happens to
  be in at the instant it emits, because different bond lengths/angles put you at
  different points on both the S1 and S0 surfaces (and hence a different energy gap
  between them). A real fluorescence spectrum is not one sharp line but a band with
  vibrational structure and width, because it's really a statistical mixture of these
  slightly different instantaneous transitions, weighted by how likely each geometry is
  to be visited.
- **Inhomogeneous broadening:** by sampling a whole distribution of excited-state
  geometries (rather than one point), and computing the emission energy at each, you get a
  distribution of transition energies. Averaging (technically, convolving) over that
  distribution reproduces the natural *width* of the experimental band — something a
  single vertical calculation structurally cannot do, since one point has no width at all.

Dynamics adds something a static optimization can't: it lets correlated, anharmonic
motion — multiple bonds/angles moving together in ways that aren't just independent,
symmetric vibrations — show up naturally in the resulting geometry ensemble, because the
trajectory is generated by the real (not simplified) forces on the real potential energy
surface.

---

## 5. How TD-DFT forces/gradients work, and what "excited-state gradient" means

To move atoms step by step in a simulation, you need the **force** on each atom, and
force is just the negative gradient of the energy with respect to nuclear position:
F = −dE/dR. For ground-state DFT, this gradient is standard and well established. For an
excited state computed via **TD-DFT** (time-dependent DFT, or its Tamm–Dancoff
approximation, TDA), the excited-state energy is not simply "the ground-state energy plus
a constant shift" — it comes from solving a separate linear-response eigenvalue problem
on top of the ground-state electron density. So the excited-state gradient has to account
for how *both* the ground-state density *and* the excitation itself change as the nuclei
move — this is more expensive to compute than a ground-state gradient (implemented via a
specialized formalism, sometimes called "response gradients" in the TD-DFT literature),
but the physical meaning is identical to any other gradient: it's the slope of the
excited-state PES at the current geometry, and following it downhill is exactly how a
geometry optimization finds the excited-state minimum (Step 3), and how molecular
dynamics decides which direction to push the atoms at each timestep (Step 4).

The key conceptual point: **"gradient in the excited state"** means the force is computed
from the shape of the *excited-state* PES, not the ground state's. Two atoms sitting at
the exact same geometry can feel completely different forces depending on which
electronic state (which PES) you're asking about — that's precisely why the S1 minimum
geometry (Step 3) is different from the S0 minimum geometry (Step 1): the two surfaces
have their low points in different places, because a different electron distribution
pulls differently on the same nuclei.

---

## 6. What the PCM solvent model does, and equilibrium vs. non-equilibrium solvation

**PCM** (Polarizable Continuum Model) approximates the solvent not as individual explicit
molecules, but as a structureless dielectric continuum surrounding a cavity shaped like
the solute. The solute's charge distribution polarizes this continuum, and the polarized
continuum in turn creates an electric field (a "reaction field") that acts back on the
solute — shifting its energy levels, and therefore its absorption/emission energies,
relative to vacuum. This is why the same molecule can absorb/emit at meaningfully
different wavelengths in different solvents (e.g., water vs. ethanol vs. vacuum): the
polarity of the surrounding medium physically shifts the electronic energy gaps.

Why treat it in *both* the ground and excited state, not just one: since the solute's
electron distribution is different in S0 versus S1 (that's the whole reason there's an
excited-state PES that's shaped differently from the ground state's), the solvent
polarizes differently around each state too. If you optimize S0 with a solvent-relaxed
reaction field but then evaluate S1 in vacuum (or vice versa), you get an internally
inconsistent picture where the solvent "helps" one state and not the other, distorting the
computed energy gap.

The **equilibrium vs. non-equilibrium** distinction matters specifically at the moment of
a *vertical* transition. A dielectric medium's polarization has two components with very
different response times: fast **electronic polarization** (essentially instantaneous —
it just needs the solvent's own electrons to shift slightly) and slow **orientational/
nuclear polarization** (solvent molecules physically reorienting, which takes much longer,
comparable to or slower than nuclear vibration). Because a vertical transition happens
faster than solvent molecules can reorient (same Franck–Condon logic as Section 2, applied
to the solvent instead of the solute), the physically correct picture for an *instantaneous*
vertical transition is **non-equilibrium solvation**: only the fast electronic part of the
solvent polarization responds during the jump, while the slow part stays frozen in
whatever configuration it had equilibrated to for the *starting* state. **Equilibrium
solvation**, by contrast, assumes the *entire* solvent polarization (fast and slow) has
fully relaxed to the current electronic state — which is the physically correct treatment
for a geometry sitting at rest at a minimum (S0 or S1), or for an MD snapshot that's had
time to thermally equilibrate, but is an approximation (arguably too generous to the
solvent) when applied to an instantaneous vertical jump.

---

## 7. Building the final spectrum from many geometries: broadening and averaging

A single vertical calculation gives one energy and one intensity (oscillator strength) —
a single infinitely sharp line, which no real spectrum looks like. Real spectra are bands
with width and shape, for two combined reasons: (a) each individual molecule's transition
already has some natural linewidth from quantum-mechanical effects (lifetime broadening,
individual vibronic transitions), and (b) at any instant, different molecules in a real
sample are sitting in slightly different geometries/environments, so they don't all emit
at exactly the same energy — the ensemble-averaged spectrum is a *superposition* of many
slightly different individual transition energies.

The trajectory-based (or ensemble-based) approach reproduces effect (b) directly: many
geometries are sampled (from an MD trajectory, or from a static distribution), and at
each one the vertical transition energy and oscillator strength (intensity) are computed
independently — as if you had many differently-configured "snapshots" of the molecule,
each contributing its own line. To turn this collection of discrete lines ("stick
spectrum") into something resembling an experimental curve, each stick is broadened —
typically by convolving each computed energy with a Gaussian (or Lorentzian) of some width
σ, and summing:

  I(E) = Σᵢ wᵢ · exp[−(E − Eᵢ)² / 2σ²]

where wᵢ is a weight related to that geometry's oscillator strength (how strongly that
particular configuration would emit). Summing many such Gaussians, one centered at each
sampled geometry's transition energy, produces a smooth band whose width reflects both the
individual line's natural width (σ) *and* the spread of transition energies across the
sampled ensemble (item (b) above) — this second contribution is called **inhomogeneous
broadening**, and it's exactly what a single vertical calculation is fundamentally
incapable of producing, no matter how accurate that one calculation is, because it never
sees more than one geometry.

Averaging over many geometries (rather than trusting one) is therefore not just "more
data for robustness" — it's supplying an entire physical ingredient (the spread from
real nuclear motion) that a single point calculation structurally omits from the model.

---

## 8. Tier 1 vs. Tier 2 vs. Tier 3 — what each one relaxes, and who benefits from going higher

| | What it computes | What it assumes away |
|---|---|---|
| **Tier 1** — static vertical | One transition energy/intensity, at one fixed geometry (typically the relaxed S1 minimum for emission, or the S0 minimum for absorption). | All nuclear motion / thermal spread. Implicitly treats the molecule as frozen at 0 K, at exactly its equilibrium geometry. |
| **Tier 2** — Nuclear Ensemble Approach (e.g. Wigner sampling) | Many transition energies/intensities, at many geometries drawn from a *static*, harmonic distribution around the relaxed minimum (built from that minimum's vibrational normal modes and frequencies — a quantum zero-point/thermal distribution, but constructed once, not evolved in time). | The distribution is approximated as harmonic (quadratic PES near the minimum) and as a set of *independent* normal modes — no anharmonicity, no mode–mode coupling, no dynamical/temporal correlation between successively sampled geometries, and (for the excited state) no coupling between the solvent's actual relaxation and the nuclear motion. |
| **Tier 3** — full excited-state AIMD | Many transition energies/intensities, at many geometries sampled from an *actual, real-time trajectory* propagated on the true (not harmonically approximated) excited-state PES, at finite temperature, with real inter-mode coupling and (if included) real-time solvent response. | Comparatively little — this is the most complete of the three; its main remaining approximations are the accuracy of the underlying electronic-structure method itself (functional, basis set) and practical limits on trajectory length/statistics, not structural approximations to the sampling model. |

Each tier "relaxes" the assumption the previous tier baked in: Tier 2 relaxes Tier 1's
"nuclei are frozen" assumption by adding a static distribution; Tier 3 relaxes Tier 2's
"the distribution is harmonic and time-independent" assumption by generating it from real
dynamics instead.

**Which molecules benefit most from going up a tier:** the gain from Tier 1 → Tier 2 is
largest for molecules with **any meaningful vibrational broadening or Stokes shift** —
essentially anything real, since a bare vertical number never has a band shape to compare
against experiment. The gain from Tier 2 → Tier 3 specifically is largest for molecules
that are **floppy, have low-frequency/large-amplitude motions, exhibit significant
anharmonicity, or have excited-state processes that couple strongly to the environment in
a time-dependent way** (e.g., twisted intramolecular charge transfer, conformational
changes upon excitation, strong coupling to slow solvent reorganization). A **rigid,
planar molecule with a stiff excited-state minimum and small-amplitude, largely harmonic
vibrations** — one whose nuclear motion is well described by a handful of independent,
near-quadratic normal modes — is exactly the case where Tier 2's harmonic approximation is
already close to reality, so the additional physical realism Tier 3 adds (anharmonicity,
mode coupling, real dynamical solvent response) has comparatively little room to change
the answer, even though it is, in principle, the more rigorous method.
