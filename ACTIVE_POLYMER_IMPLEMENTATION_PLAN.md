# ActivePolyBD — Design & Recap Document

**A Brownian-dynamics simulator for tangentially active polymers.**

**Target model:** Bianco, Locatelli & Malgaretti, *Globulelike Conformation and Enhanced Diffusion of Active Polymers*, Phys. Rev. Lett. **121**, 217802 (2018), plus its Supplemental Material.

**Status:** as-built. Phases 0–6 complete; the package precompiles, **160/160 tests pass**, and the headline P(R_G)-vs-Pe result reproduces the coil→globule trend. Phase 7 (Verlet lists, GPU batching, cone activity α>0) is intentionally left as future work. Built and tested on Julia 1.12.

> This document began life as a forward-looking implementation plan. It has been rewritten to describe **what was actually built** and to record the decisions and deviations made along the way. Section 2 (physics) remains the normative spec — it was implemented exactly as stated. Sections 11 (decisions/deviations) and 10 (validation results) are new and capture how the delivered code differs from the original plan and how we know it is correct.

---

## 1. Scope delivered

**What it does**
1. Overdamped Brownian dynamics (Rouse regime: no hydrodynamics, no periodic box) of a single bead–spring chain in unbounded ℝ³.
2. Tangential active force on interior monomers only (α = 0 exactly; ends passive).
3. Reduced (dimensionless) units throughout.
4. Two equivalent scriptable entry points: a **TOML config file** and a thin **Julia API** — a run is fully specified by an input script.
5. User-settable timestep, equilibration/production step counts, save/log frequencies, seed, XYZ output, checkpoint cadence.
6. Extensible bonded and non-bonded interactions via Julia multiple dispatch: adding e.g. a FENE bond or a WCA core is "one struct + its methods + one registry line," with no edits to the integrator or force loop. `FENEBond`, `WCA`, and `NoActivity` ship as worked examples.
7. R_G computed on the fly during production and logged; the headline deliverable is the **R_G probability distribution P(R_G) for varying Pe** (fixed N, e.g. N=40) overlaid on one plot.
8. A correct, tested engine — validated with cheap analytic sanity checks (Section 10), not a scaling-exponent study.

**Out of scope (by design; clean extension points left in place)**
- Cone activity α > 0 — the `ActivityModel` interface admits it; not implemented (see `activity.jl` note).
- Hydrodynamic interactions.
- Periodic boundaries / explicit solvent.
- GPU — not needed for N ≤ 40; a batched path is future work.
- Scaling exponents ν(Pe), N-sweeps, D_long fits, phase diagrams — the headline output is P(R_G) vs Pe at fixed N, not these.

---

## 2. Physics specification (normative — implemented exactly)

### 2.1 Equation of motion

Overdamped Langevin / Brownian dynamics for each monomer i:

```
ṙ_i = β D₀ ( −∇_i V + f_i^act ) + η_i
```

with Gaussian white noise obeying the fluctuation–dissipation relation

```
⟨ η_l(t) η_k(t′) ⟩ = 2 D₀ δ_lk δ(t − t′),      β ≡ 1/(k_B T),   D₀ = k_B T / ζ.
```

There is **no simulation box**: the chain lives in unbounded ℝ³ and its center of mass drifts freely. Dimensionality d = 3.

### 2.2 Interactions

**Bonded (harmonic springs, backbone neighbors i−1, i+1):**

```
V_i^sp = Σ_{j=i−1,i+1} (K_sp/2) (r_ij − b)²
```

**Non-bonded (purely repulsive, one-sided harmonic; only non-backbone-neighbor pairs, and only when overlapping r_ij < b):**

```
V_i^mm = Σ_{j≠i−1,i+1} (K_sp/2) (r_ij − b)²      for r_ij < b   (zero otherwise)
```

Same parabola and same stiffness `K_sp` as the bond, but **one-sided (purely repulsive)**: the bond potential is *two-sided* — it restores in both directions — whereas the non-bonded interaction keeps **only the repulsive half of the parabola**, active for r < b and **identically zero for r ≥ b**. At r = b both `V` and `V′ = K_sp(r−b)` go to zero, so the force turns on continuously from contact (C¹). Self-avoidance is therefore **soft** — beads can overlap under strong forcing; `K_sp = 100` makes crossings rare, not impossible.

**Stiffness:** `K_sp = 100 k_B T / b²`.
> The paper writes `K_sp = 100 k_B T / b`, but `(K_sp/2)(r−b)²` requires energy/length² for the energy to come out in k_B T — a units typo. In reduced units the numerical value is **100** either way, so the simulator is unaffected; documented so no one "fixes" it.

*Implemented as:* `HarmonicBond(k, r0)` (two-sided) and `SoftRepulsive(k, cutoff)` (one-sided, `r < cutoff` only). See `src/potentials_bonded.jl`, `src/potentials_pair.jl`.

### 2.3 Active force (α = 0 only)

For interior monomers i = 2 … N−1 (1-based indexing), with the "skip-one" tangent vector `t_i = r_{i+1} − r_{i−1}`:

```
f_i^act = f^act · t_i / |t_i|
```

The **first and last monomers (i = 1, i = N) are passive**: `f_i^act = 0`.

The active force is **non-conservative** — *not* the gradient of any potential. It is added directly to the force accumulator. Potential energy is therefore not a meaningful stability check; bond-length and R_G stability are used instead (Section 10).

*Implemented as:* `TangentialActivity(Pe)` in `src/activity.jl`; `add_active_forces!` loops `i = 2:N-1` only.

### 2.4 Péclet number

```
Pe ≡ f^act b / (k_B T)   =   v^act b / D₀
```

### 2.5 Reduced units (used throughout)

| Quantity            | Unit                  | Reduced value              |
|---------------------|-----------------------|----------------------------|
| length              | b (monomer diameter)  | b = 1                      |
| energy              | k_B T                 | k_B T = 1  ⇒ β = 1         |
| time                | τ_B = b² / D₀         | τ_B = 1  ⇒ D₀ = 1          |
| mobility β D₀       | 1/ζ                   | β D₀ = 1                   |
| stiffness K_sp      | k_B T / b²            | 100                        |
| active force magnitude f^act | k_B T / b    | **= Pe**                   |
| bond rest length r₀ | b                     | 1                          |
| timestep dt         | τ_B                   | default 1e-3               |

Because f^act (reduced) = Pe, the active term in the update is simply `Pe · t̂_i`.

### 2.6 Integrator — Euler–Maruyama (reduced units)

```
r_i ← r_i + ( F_i^cons + F_i^act ) · dt + sqrt(2·dt) · ξ_i
```

where `F_i^cons = −∇_i V` (bonded + non-bonded), `F_i^act = Pe·t̂_i` (interior) or 0 (ends), and `ξ_i` is a vector of d independent standard normals `N(0,1)` drawn fresh each step per component.

- Default `dt = 1e-3`; exposed as a parameter, with a dt-convergence test in the suite.
- Shared pairwise force convention (all pair/bond geometry lives in one place). With `r̂_ij = (r_i − r_j)/r_ij`:

```
F_i (pair i,j) = −V′(r_ij) · r̂_ij ,     F_j = +V′(r_ij) · r̂_ij
```

  Each potential supplies only `V′(r)` (= `dVdr`), and optionally `energy(r)` and `cutoff`. Sign checks: harmonic bond `V′ = K(r−r₀)` (restoring when stretched); soft-repulsive `V′ = K(r−b)` for r<b (< 0 ⇒ pushes apart).

*Implemented as:* `bd_step!(sys, dt, rng)` in `src/integrator.jl`; the shared geometry rule lives in `pair_force` in `src/forces.jl`.

### 2.7 Analytic reference results (used by the tests, not inside the engine)

- Free single particle: MSD = 2·d·D₀·t = 6t (3D, reduced) — checks the noise amplitude.
- Passive-chain CM diffusion: `D_short = D₀/N = 1/N` — checks mobility and 1/N free-draining scaling. (In this model the CM decouples exactly, so this holds at all times, not just short times — see Section 10.)

Qualitative expectation for the headline result: **mean R_G decreases and P(R_G) shifts toward smaller values and sharpens as Pe increases** (coil → globule-like), per Fig. 2a–b. Confirmed (Section 10).

---

## 3. Language and architecture

**Language: Julia.** For N ≤ 40 there is no box and no HI, so per-step cost is trivial (~N²/2 ≈ 800 pairs). The dominant cost is the huge step count × independent replicas — the problem is *embarrassingly parallel over replicas*, and what matters is (a) a cheap, allocation-free inner loop and (b) easy multithreading across replicas. Neither GPU nor neighbor lists are on the critical path at this size.

Requirement 6 (extensible interactions) maps directly onto Julia's multiple dispatch: a new potential is a new `struct` plus a `dVdr`/`energy` method; the integrator and force loop never change and stay type-stable and inlined. The `System` type is concretely parameterized on its potential/activity/neighbor types so every force call specializes — this type stability is the single biggest performance lever, and it was verified: `bd_step!` allocates **0 bytes** (Section 10).

Module boundaries were chosen so a Python (Numba) port would be a near 1:1 mapping, though Julia is the delivered and recommended implementation.

---

## 4. Repository layout (as built)

The package **is the repository root** (`ActivePolyBD/`); there is no nested package directory. The module is `ActivePolyBD`, defined in `src/ActivePolyBD.jl`.

```
ActivePolyBD/                       # repo root = package home
├── Project.toml                    # name = "ActivePolyBD"; deps (Section 15)
├── Manifest.toml                   # resolved deps (generated)
├── README.md                       # quickstart, unit conventions, extension recipe
├── ACTIVE_POLYMER_IMPLEMENTATION_PLAN.md   # this design/recap document
├── src/
│   ├── ActivePolyBD.jl             # module root: includes + exports
│   ├── types.jl                    # System, SimParams, Vec3, deepcopy_system
│   ├── potentials_bonded.jl        # BondedPotential; HarmonicBond; FENEBond (example)
│   ├── potentials_pair.jl          # PairPotential; SoftRepulsive; WCA (example)
│   ├── activity.jl                 # ActivityModel; TangentialActivity; NoActivity
│   ├── neighbors.jl                # NeighborStrategy; AllPairs; VerletList (stub); bonded_exclusion
│   ├── forces.jl                   # compute_forces!: bonded + pair + active
│   ├── integrator.jl               # Euler–Maruyama bd_step!
│   ├── observables.jl              # Rg, gyration tensor, eigenvalues, asphericity, Re, angles, CM
│   ├── io.jl                       # XYZ writer, scalar log, Serialization checkpoint/restart
│   ├── config.jl                   # TOML → RunConfig; potential/activity registries
│   ├── init.jl                     # chain initialization (:line | :saw)
│   └── simulation.jl               # run_single!; threaded replica driver run!
├── scripts/
│   ├── run.jl                      # julia -t auto scripts/run.jl config.toml
│   ├── sweep.jl                    # Pe sweep → per-Pe subdirs → analysis
│   ├── bench.jl                    # steps/s + allocation micro-benchmark
│   ├── example.toml                # documented single-run config
│   └── sweep.toml                  # documented sweep config
├── analysis/
│   └── rg_distribution.jl          # read logs → P(R_G) overlay + ⟨R_G⟩-vs-Pe table (module RgDistribution)
└── test/
    ├── runtests.jl                 # includes the six suites below
    ├── test_potentials.jl          # force↔energy finite-difference (all 4 potentials)
    ├── test_forces_activity.jl     # force balance, exclusions, passive ends, skip-one tangent
    ├── test_integrator.jl          # free MSD=6t; passive CM D=1/N; determinism; dt convergence
    ├── test_observables.jl         # gyration identities, asphericity, baseline bond lengths
    ├── test_config_io.jl           # config parsing/validation, XYZ/log, checkpoint restart
    └── test_headline.jl            # ⟨R_G⟩ decreases with Pe (qualitative check 8)
```

---

## 5. Core module reference (as implemented)

Per-monomer vectors use `StaticArrays.SVector{3,Float64}` (aliased `Vec3`); positions/forces are `Vector{Vec3}` of length N.

### 5.1 `types.jl`

```julia
const Vec3 = SVector{3,Float64}

struct System{B<:BondedPotential, P<:PairPotential, A<:ActivityModel, NS<:NeighborStrategy}
    N::Int
    positions::Vector{Vec3}
    forces::Vector{Vec3}          # preallocated scratch, overwritten each step
    bonded::B
    pair::P
    activity::A
    neighbors::NS
end

# Convenience keyword constructor (allocates the force buffer):
System(positions; bonded, pair, activity, neighbors=AllPairs())

struct SimParams
    dt; n_equil; n_prod; xyz_every; log_every; checkpoint_every; seed::UInt64
end
# Keyword constructor with defaults:
SimParams(; dt=1e-3, n_equil=1_000_000, n_prod=10_000_000,
            xyz_every=1000, log_every=1000, checkpoint_every=1_000_000, seed=12345)

deepcopy_system(sys)   # independent copy with fresh position/force buffers
```

`System` is concretely parameterized so all force calls specialize and inline.

### 5.2 Bonded potentials — `potentials_bonded.jl`

```julia
abstract type BondedPotential end

struct HarmonicBond <: BondedPotential; k; r0; end
dVdr(p::HarmonicBond, r)   = p.k*(r - p.r0)
energy(p::HarmonicBond, r) = 0.5*p.k*(r - p.r0)^2

struct FENEBond <: BondedPotential; k; R0; end        # extension example
dVdr(p::FENEBond, r)   = p.k*r / (1 - (r/p.R0)^2)
energy(p::FENEBond, r) = -0.5*p.k*p.R0^2*log(1 - (r/p.R0)^2)
```

### 5.3 Pair (non-bonded) potentials — `potentials_pair.jl`

```julia
abstract type PairPotential end

struct SoftRepulsive <: PairPotential; k; cutoff; end     # the paper's one-sided core
cutoff(p::SoftRepulsive) = p.cutoff
dVdr(p::SoftRepulsive, r)   = r < p.cutoff ? p.k*(r - p.cutoff) : 0.0
energy(p::SoftRepulsive, r) = r < p.cutoff ? 0.5*p.k*(r - p.cutoff)^2 : 0.0

struct WCA <: PairPotential; eps; sigma; end              # extension example
cutoff(p::WCA) = 2^(1/6)*p.sigma                          # cut & shifted at the minimum
```

**Exclusions:** the pair loop skips backbone-bonded neighbors via `bonded_exclusion(i,j) = abs(i-j)==1`, defined in `neighbors.jl` — a property of the neighbor strategy, not hard-coded in any potential.

### 5.4 Activity — `activity.jl`

```julia
abstract type ActivityModel end
struct NoActivity <: ActivityModel end
struct TangentialActivity <: ActivityModel; Pe; end

add_active_forces!(forces, pos, ::NoActivity) = forces
function add_active_forces!(forces, pos, a::TangentialActivity)
    for i in 2:length(pos)-1                     # interior only; ends passive
        t = pos[i+1] - pos[i-1]                  # skip-one tangent
        forces[i] += a.Pe * (t / norm(t))
    end
end
```

A comment marks where a future `ConeActivity{α}` (per-monomer orientation state, rotational diffusion + cone reflection, SM Eq. 2) would slot in as another `ActivityModel`.

### 5.5 Neighbors — `neighbors.jl`

```julia
abstract type NeighborStrategy end
bonded_exclusion(i, j) = abs(i - j) == 1
struct AllPairs <: NeighborStrategy end          # default: i<j double loop, exclusion + cutoff
struct VerletList <: NeighborStrategy; skin; end # interface-complete stub
```

`AllPairs` is fastest for N ≤ 40. `VerletList` currently **delegates to `AllPairs` semantics** (identical results); the rebuild-on-max-displacement logic is Phase 7.

### 5.6 Forces — `forces.jl`

```julia
compute_forces!(sys) =
    fill!(sys.forces, zero(Vec3));
    add_bonded_forces!(...); add_pair_forces!(...); add_active_forces!(...)
```

All pair/bond accumulation uses the shared rule: `d = r_i − r_j`, `r = |d|`, `fmag = dVdr(pot, r)`, `f = -(fmag/r)*d`; `forces[i] += f; forces[j] -= f`. No allocations in the loops. The pair loop uses squared-distance culling against `cutoff²` before taking a square root.

### 5.7 Integrator — `integrator.jl`

```julia
function bd_step!(sys, dt, rng)
    compute_forces!(sys)
    s = sqrt(2*dt)                               # reduced units: D₀ = βD₀ = 1
    for i in 1:sys.N
        ξ = Vec3(randn(rng), randn(rng), randn(rng))
        sys.positions[i] += sys.forces[i]*dt + s*ξ
    end
end
```

**RNG:** `Random.Xoshiro`, one independent stream per replica (Section 9). Fast `randn`, never reseeded inside the loop.

### 5.8 Initialization — `init.jl`

```julia
initialize_chain(N; mode=:line|:saw, rng, b=1.0, kink=0.1)
```

- `:line` — a straight backbone along x with small random transverse kinks, bond lengths rescaled to exactly `b`.
- `:saw` — self-avoiding growth walk: each new bead placed a distance `b` away in a random direction, rejecting overlaps (r < b).

Requires N ≥ 3 so interior monomers exist.

---

## 6. Observables — `observables.jl`

All in reduced units; cheap for N ≤ 40. **R_G is computed during production and logged.**

**Primary (required):**
- `center_of_mass(pos)` = mean position.
- `gyration_tensor(pos)` = `S = (1/N) Σ_i (r_i−cm)(r_i−cm)ᵀ` (3×3 `SMatrix`).
- `radius_of_gyration(pos)` = `sqrt(tr S)`; `radius_of_gyration_sq` avoids the root.

**Optional (hooks present; delivery not gated on them):**
- `gyration_eigenvalues(pos)` → λ₁≥λ₂≥λ₃ via `eigvals(Symmetric(...))`.
- `AsphericityAccumulator` / `asphericity(acc)` = `⟨Tr² − 3M⟩ / ⟨Tr²⟩`, accumulating numerator and denominator **separately** over frames (never per-frame ratios), with `Tr = λ₁+λ₂+λ₃`, `M = λ₁λ₂+λ₁λ₃+λ₂λ₃`.
- `end_to_end(pos)` = `r_N − r_1`.
- `backbone_cosangles(pos)` = `cosθ_i = û_{i-1,i}·û_{i,i+1}` at interior monomers.

---

## 7. I/O — `io.jl`

- **XYZ trajectory** (extended-XYZ friendly): per frame `N`, a comment line `step=… time=… Rg=…`, then `element x y z` per monomer. Terminal (passive) monomers use element `O`; interior use `C`, so ends are visible in VMD/OVITO.
- **Scalar log** (`.dat`, whitespace-delimited, `#`-commented header): `step time Rg cm_x cm_y cm_z`, one row every `log_every`. This file is the input to the P(R_G) analysis.
- **Checkpoint/restart:** `save_checkpoint`/`load_checkpoint`/`restart!` snapshot `positions`, `step`, and the **full RNG state** using the `Serialization` stdlib (see Section 11 for why not JLD2). Writes go to a `.tmp` file then atomically `mv` into place so a crash can't truncate a checkpoint. `restart!` loads a checkpoint into an existing `System` and returns `(step, rng)` to continue production.
- All frequencies are user parameters.

---

## 8. Configuration / scripting interface — `config.jl`

Two equivalent entry points build the *same* objects.

**(a) TOML input script**, parsed via **registries** mapping type-strings → builder closures, so a newly added potential becomes usable from config with no parser edits:

```julia
BONDED_REGISTRY   = Dict("harmonic"=>…, "fene"=>…)
PAIR_REGISTRY     = Dict("soft_repulsive"=>…, "wca"=>…)
ACTIVITY_REGISTRY = Dict("tangential"=>…, "none"=>…)
```

`load_config(path)` / `parse_config(dict)` produce a `RunConfig` (immutable interaction models + `SimParams` + `OutputSpec` + replica settings). Per-replica `System`s are built lazily by `build_system(cfg, rng)` so each replica gets an independent initial configuration.

Validation on load (with clear error messages): positive `dt`, `N ≥ 3`, `dimensions == 3`, known type strings, non-negative `Pe`, `n_replicas ≥ 1`, and `parallel ∈ {serial, threads}`.

**(b) Julia API:** build `System` + `SimParams` + `OutputSpec` directly and call `run_single!` (or build a `RunConfig` and call `run!`).

### 8.1 Example `example.toml` (single run)

```toml
[system]
N = 40; dimensions = 3; seed = 12345; init = "line"   # or "saw"

[interactions.bonded]
type = "harmonic"; k = 100.0; r0 = 1.0

[interactions.nonbonded]
type = "soft_repulsive"; k = 100.0; cutoff = 1.0      # = b

[activity]
type = "tangential"; Pe = 10.0                        # = 0 or type="none" for passive

[integrator]
dt = 1e-3; equilibration_steps = 100_000; production_steps = 1_000_000

[output]
xyz_file = "traj.xyz"; xyz_every = 5000
log_file = "obs.dat";  log_every = 1000
checkpoint_file = "checkpoint.jls"; checkpoint_every = 500_000

[replicas]
n_replicas = 1; parallel = "serial"                   # "serial" | "threads"
```

(TOML tables are shown compacted here; the shipped file uses one key per line.)

---

## 9. Simulation driver and replicas — `simulation.jl`

- `run_single!(sys, params, rng, output; replica_id, restart_path, collect_rg)`: equilibrate `n_equil` steps (no data written), then run `n_prod` production steps writing XYZ/log/checkpoints at their strides; returns a `ReplicaResult` (in-memory R_G series + written paths). The CM is **never recentered** during production — its free drift is physical. If `restart_path` points to a checkpoint, positions/step/RNG are loaded and production continues (equilibration skipped, output files appended).
- `run!(cfg::RunConfig)`: builds and runs `n_replicas` independent replicas. Each gets its own freshly initialized `System`, its own RNG stream, and output files suffixed `_r<id>` (`traj.xyz` → `traj_r3.xyz`). `parallel="threads"` → `Threads.@threads` over replicas (start Julia with `-t auto`); `"serial"` runs them in order.
- **Per-replica RNG:** `replica_rng(seed, id) = Xoshiro(seed ⊻ (0x9e3779b97f4a7c15 * id))` — a well-separated, independent stream per replica. Same `(seed, id)` reproduces bit-for-bit; distinct ids give distinct trajectories (both verified).

---

## 10. Validation and results (how we know it's correct)

All checks run in `test/` (`julia --project=. -e 'using Pkg; Pkg.test()'`). **Result: 160 pass, 0 fail.** We deliberately do **not** validate scaling exponents, D_long, correlation times, or phase diagrams.

**Unit-level:**
1. **Force ↔ energy consistency** — analytic `dVdr` matches a central finite difference of `energy(r)` for all four potentials (`HarmonicBond`, `FENEBond`, `SoftRepulsive`, `WCA`), rel. err < 1e-6. Also checks the `SoftRepulsive` C¹ turn-on (force and energy vanish at/above cutoff).
2. **Free single particle** (N=1) — MSD(t) = 6t within ~5% over 4000 walkers. Verifies the `sqrt(2·dt)` noise amplitude and mobility = 1.
3. **Passive CM diffusion** — CM MSD = 6·t/N within ~6%. In this free-draining model the internal (bonded + pair) forces cancel in the CM by Newton's third law, so the CM performs a pure random walk with `D_cm = 1/N` **exactly at all times**; the test exploits this.
4. **Gyration-tensor identities** — `tr(S) == R_G²`, symmetric S, non-negative ordered eigenvalues, and a known two-bead configuration. Plus asphericity = 1 for a rod and 0 for an isotropic cloud.
5. **Determinism & replica independence** — same seed ⇒ identical trajectory; different replica id ⇒ different trajectory.
6. **dt convergence** — ⟨R_G⟩ stable between dt=1e-3 and dt=5e-4 at matched physical time.
7. **Force-loop correctness** — conservative internal forces sum to zero; the pair loop excludes bonded neighbors and respects the cutoff; active force is zero on the ends, has magnitude `Pe` on interior monomers, and points along the skip-one tangent.
8. **Config / I/O** — registry parsing (incl. FENE/WCA reachable purely through config), validation errors, XYZ element labels + log columns, and a **checkpoint round-trip** proving a restart reproduces the continuation bit-for-bit.
9. **Passive baseline** — ⟨(r_ij−b)²⟩ ≈ k_BT/K_sp = 1/100 per bond (harmonic equipartition), confirming equilibration.

**Headline result (qualitative check 8): confirmed.** Both the in-suite trend test and a standalone sweep show ⟨R_G⟩ decreasing and P(R_G) sharpening as Pe rises (coil → globule):

| Pe | ⟨R_G⟩ | std |
|----|-------|-----|
| 0  | 4.41  | 0.79 |
| 5  | 3.27  | 0.65 |
| 50 | 2.22  | 0.26 |

(N=40, short verification run; longer runs sharpen the tails.) A separate N=20 test gives ⟨R_G⟩ 2.60 (passive) → 1.36 (Pe=100).

**Performance:** `scripts/bench.jl` reports **~1.15×10⁶ steps/s** for an N=40 active chain single-threaded, with **0 bytes allocated per `bd_step!`** — confirming the type-stable, allocation-free hot loop. Replica parallelism scales this across cores.

---

## 10a. Headline analysis: P(R_G) vs Pe

The one scientific deliverable. Workflow (`scripts/sweep.jl` + `scripts/sweep.toml`):
1. Run the same system (fixed N, dt, equilibration/production lengths) at several Pe values — default `Pe ∈ {0, 0.1, 1, 5, 10, 50}` — with a few replicas each for smoother histograms. Each Pe gets its own subdirectory `<out_dir>/pe_<value>/` containing per-replica logs and a `pe.txt` label.
2. Each run logs R_G every `log_every` steps during production.
3. `analysis/rg_distribution.jl` (module `RgDistribution`) reads all logs, discards a warm-up fraction, histograms R_G per Pe on a shared grid, and writes:
   - `pRg_vs_Pe.csv` — bin centers + P(R_G) density per Pe,
   - `mean_Rg_vs_Pe.csv` — ⟨R_G⟩, std, and sample count per Pe,
   - an ASCII overlay + summary table to stdout,
   - `pRg_vs_Pe.png` **only if** the optional `Plots` package is installed (see Section 11).
4. `scripts/sweep.jl` runs the whole sweep and then the analysis from one command.

Samples are time-correlated, so the histogram is treated as a density estimate; more replicas or longer production simply smooth the tails.

---

## 11. Decisions and deviations from the original plan

This section records where the delivered code intentionally differs from the initial plan.

1. **Package name & layout.** The package is `ActivePolyBD` and lives at the **repository root** (flat), not a nested `ActivePolymer/` directory as originally sketched. The module is `ActivePolyBD` in `src/ActivePolyBD.jl`. (The plan file keeps its original `ACTIVE_POLYMER_IMPLEMENTATION_PLAN.md` name, referenced from the module docstring.)

2. **Checkpointing uses `Serialization`, not JLD2.** The plan allowed "binary or JLD2." We chose the `Serialization` stdlib to avoid a heavyweight dependency; it snapshots positions, step, and full RNG state, which is all the restart needs. Swapping in JLD2 later is a localized change in `io.jl`.

3. **Plotting is optional and out of the dependency set.** The analysis always emits CSVs + an ASCII overlay (so it runs with zero extra deps) and produces a PNG only if `Plots` happens to be installed. This keeps `Pkg.instantiate` lightweight — `StaticArrays` is the only non-stdlib dependency. Add `Plots`/`CairoMakie` if a PNG is wanted by default.

4. **RNG is `Xoshiro`, not `Random123`/Philox.** The plan offered either. `Xoshiro` (Julia's default, fast `randn`) with a golden-ratio-mixed per-replica seed gives well-separated, reproducible streams, verified by the determinism/independence test. `Random123` can be substituted if strict cross-platform stream identity is ever required.

5. **More test files than the three originally listed.** Delivered suites: `test_potentials.jl`, `test_forces_activity.jl` (new), `test_integrator.jl`, `test_observables.jl`, `test_config_io.jl` (new), `test_headline.jl` (new). The extra suites cover the force loop/exclusions/activity, config+I/O+restart, and the qualitative headline trend as an automated check.

6. **`VerletList` delegates to `AllPairs`.** Shipped as an interface-complete stub returning identical results, so it can be selected without changing behavior; the displacement-triggered rebuild is Phase 7. `bonded_exclusion` lives in `neighbors.jl` as a free function.

7. **`RunConfig` + lazy per-replica system build.** Config parsing yields a `RunConfig` holding immutable interaction models; `build_system(cfg, rng)` constructs each replica's `System` on demand with its own RNG, giving independent initial conditions. `OutputSpec` carries the three output file bases (including `checkpoint_file`).

8. **Convenience constructors added.** `System(positions; bonded, pair, activity, neighbors)` and a keyword `SimParams(; …)` with defaults, alongside the raw positional constructors used internally.

9. **Build note (Julia 1.12).** During bring-up the `Printf` stdlib UUID in `Project.toml` had to match this Julia's value (`de0858da-…-51eddeeeb8d7`). Not a design choice, but recorded so a future environment change doesn't reintroduce a stale UUID.

Nothing in the normative physics (Section 2) was changed — signs, index ranges, the skip-one tangent, passive ends, the one-sided non-bonded core, and the `sqrt(2·dt)` noise amplitude are all implemented exactly as specified.

---

## 12. Implementation phases (all complete except Phase 7)

- **Phase 0 — Scaffold.** ✅ Package, `Project.toml`, module compiles, `Pkg.test()` runs.
- **Phase 1 — Interactions + forces.** ✅ Types, `HarmonicBond`, `SoftRepulsive`, `AllPairs`, exclusions, `compute_forces!`; force↔energy test passes.
- **Phase 2 — Integrator + baseline physics.** ✅ `bd_step!`, RNG, init; free-particle MSD=6t, passive D=1/N, determinism pass.
- **Phase 3 — Activity + run loop + I/O + config.** ✅ `TangentialActivity`, `run_single!`, XYZ/log/checkpoint, TOML + registries, `scripts/run.jl`; restart reproduces continuation.
- **Phase 4 — Observables + R_G logging.** ✅ Gyration tensor + R_G (required) wired into the scalar log; optional asphericity/R_E/angle hooks present.
- **Phase 5 — Replicas + parallelism.** ✅ Independent replica driver, per-replica seeds/outputs, threaded execution.
- **Phase 6 — Headline result: P(R_G) vs Pe.** ✅ `scripts/sweep.jl` + `sweep.toml` + `analysis/rg_distribution.jl`; overlay shows P(R_G) shifting to smaller R_G / sharpening as Pe increases.
- **Phase 7 — Optional / future.** ⬜ `VerletList` rebuild for larger N; batched GPU driver (`CUDA.jl`, replicas as the batch dimension); `ConeActivity` (α>0) with rotational diffusion + cone reflection (SM Eq. 2).

---

## 13. Extension guide

*Add a bond type:* define `struct MyBond <: BondedPotential`, implement `dVdr`/`energy`, register `"mybond" => d -> MyBond(d["k"])` in `BONDED_REGISTRY`. Nothing in the integrator or force loop changes. Pair potentials additionally define `cutoff(p)`; activity models define `add_active_forces!(forces, pos, ::MyActivity)`. `FENEBond`, `WCA`, and `NoActivity` are the worked examples. This is the concrete payoff of the extensibility requirement.

---

## 14. Pitfalls (each has burned a BD reproduction before) — all handled

- **Noise amplitude** is `sqrt(2·D₀·dt)` per Cartesian component; in reduced units `sqrt(2·dt)`. Off-by-√2 silently rescales temperature. (Guarded by the free-particle MSD=6t test.)
- **Ends are passive**; the tangent uses **skip-one** neighbors `r_{i+1}−r_{i−1}`, not adjacent bonds; interior range i=2…N−1 (1-based). (Guarded by the activity test.)
- **Non-bonded is one-sided** (only r<b) and **excludes bonded neighbors** (|i−j|=1). (Guarded by the pair-loop test.)
- **Active force is non-conservative** — added to forces, never to a potential; energy is not a stability check.
- **No box / no PBC / no minimum image.** The CM drifts freely and is never recentered during production.
- **Asphericity** accumulates ⟨Tr²⟩ and ⟨3M⟩ separately — never the per-frame ratio.
- **K_sp units:** value 100 in reduced units; the paper's "k_BT/b" is a typo for "k_BT/b²" (harmless numerically).
- **Reproducible parallel RNG:** each replica gets an independent, well-separated `Xoshiro` stream; no shared global RNG across threads.

---

## 15. Environment / dependencies

Julia 1.12 (compat declares ≥ 1.9). Dependencies actually used:

- **Runtime:** `StaticArrays` (only non-stdlib dep) plus stdlibs `LinearAlgebra`, `Random`, `Printf`, `Statistics`, `TOML`, `Serialization`.
- **Tests:** `Test` (stdlib).
- **Optional, not declared:** `Plots` (or another backend) for the P(R_G) PNG; the analysis degrades gracefully to CSV/ASCII without it.
- **Future (Phase 7):** `CUDA` for the batched GPU path.

Run threaded with `julia -t auto`. Quickstart:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # from the repo root
julia --project=. -e 'using Pkg; Pkg.test()'
julia -t auto --project=. scripts/run.jl   scripts/example.toml
julia -t auto --project=. scripts/sweep.jl scripts/sweep.toml
julia       --project=. scripts/bench.jl
```
