# ActivePolyBD

Overdamped Brownian-dynamics simulator for **tangentially active bead–spring
polymers** (α = 0), in reduced units, reproducing the α = 0 case of

> A. R. Bianco, E. Locatelli & P. Malgaretti, *Globulelike Conformation and
> Enhanced Diffusion of Active Polymers*, Phys. Rev. Lett. **121**, 217802 (2018).

The normative physics spec is `ACTIVE_POLYMER_IMPLEMENTATION_PLAN.md`. The
headline scientific output is the **R_G probability distribution P(R_G) versus
Péclet number** at fixed chain length N.

## Physics in one paragraph

A single chain of `N` monomers evolves by Euler–Maruyama in unbounded ℝ³ (no
box, CM drifts freely):

```
r_i ← r_i + (F_i^cons + F_i^act)·dt + sqrt(2·dt)·ξ_i
```

- `F_i^cons` = harmonic backbone bonds + a **one-sided** soft-repulsive core
  (only for overlapping non-bonded pairs, `r < b`; excludes backbone neighbors).
- `F_i^act = Pe · t̂_i` on **interior** monomers only, with the **skip-one**
  tangent `t_i = r_{i+1} − r_{i−1}`. The two terminal monomers are passive.
  The active force is **non-conservative** — never treat it as a gradient, and
  don't use energy as a stability check.
- `ξ_i` are fresh standard normals each step/component.

## Reduced units

| Quantity | Unit | Value |
|---|---|---|
| length | b | 1 |
| energy | k_BT | 1 (β = 1) |
| time | τ_B = b²/D₀ | 1 (D₀ = βD₀ = 1) |
| stiffness K_sp | k_BT/b² | 100 |
| active-force magnitude f^act | k_BT/b | **= Pe** |
| default dt | τ_B | 1e-3 |

Because `f^act = Pe` in reduced units, the active update is simply `Pe·t̂_i`.
(The paper writes `K_sp = 100 k_BT/b`; that is a units typo for `k_BT/b²` — the
numerical value 100 is unaffected. See the plan §2.2.)

## Install / test

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # from the repo root
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Run a single configuration

```bash
julia -t auto --project=. scripts/run.jl scripts/example.toml
```

Writes an extended-XYZ trajectory and a scalar `.dat` log (columns
`step time Rg cm_x cm_y cm_z`), one set per replica (files suffixed `_r<id>`).
Terminal beads are labeled `O`, interior beads `C`, so passive ends are visible
in VMD/OVITO.

## Headline result: P(R_G) vs Pe

```bash
julia -t auto --project=. scripts/sweep.jl scripts/sweep.toml
```

Runs the same chain at several Pe values (each in its own subdirectory of
`sweep_out/`), then runs `analysis/rg_distribution.jl` to write:

- `sweep_out/pRg_vs_Pe.csv` — bin centers + P(R_G) density per Pe,
- `sweep_out/mean_Rg_vs_Pe.csv` — ⟨R_G⟩, std, and sample count per Pe,
- an ASCII overlay + summary table on stdout,
- `sweep_out/pRg_vs_Pe.png` **if** the `Plots` package is installed (optional).

Expected qualitative trend: **⟨R_G⟩ decreases and P(R_G) shifts to smaller
values / sharpens as Pe increases** (coil → globule-like).

## Julia API

```julia
using ActivePolyBD, Random
rng = Xoshiro(1)
pos = initialize_chain(40; mode=:line, rng=rng)
sys = System(pos; bonded=HarmonicBond(100.0, 1.0),
             pair=SoftRepulsive(100.0, 1.0),
             activity=TangentialActivity(10.0))
params = SimParams(; dt=1e-3, n_equil=100_000, n_prod=1_000_000,
                   log_every=1000, xyz_every=5000)
run_single!(sys, params, rng, OutputSpec("traj.xyz","obs.dat","chk.jls"))
```

## Benchmark

```bash
julia --project=. scripts/bench.jl     # steps/s for N=40 + allocation check
```

## Extending interactions (req. 7)

Adding a potential is *one struct + its methods + one registry line* — the
integrator and force loop never change:

```julia
struct MyBond <: BondedPotential; k::Float64; end
@inline ActivePolyBD.dVdr(p::MyBond, r) = ...      # V'(r)
@inline ActivePolyBD.energy(p::MyBond, r) = ...    # V(r)
ActivePolyBD.BONDED_REGISTRY["mybond"] = d -> MyBond(d["k"])
```

Pair potentials additionally define `cutoff(p)`; activity models define
`add_active_forces!(forces, pos, a::MyActivity)`. `FENEBond`, `WCA`, and
`NoActivity` ship as worked examples. A `ConeActivity{α}` (α > 0) with
rotational diffusion + cone reflection is the planned next extension point
(see `activity.jl`).

## Layout

```
src/         core module (types, potentials, forces, integrator, io, config, driver)
scripts/     run.jl, sweep.jl, bench.jl, example.toml, sweep.toml
analysis/    rg_distribution.jl  (P(R_G) vs Pe)
test/        runtests.jl + unit/qualitative checks (plan §10)
```
