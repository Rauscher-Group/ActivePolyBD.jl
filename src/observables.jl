# Observables (reduced units, §6). Cheap for N ≤ 40, computed on demand.
# R_G is the primary observable and is logged during production.

"""
    center_of_mass(pos) -> Vec3
"""
@inline center_of_mass(pos::Vector{Vec3}) = sum(pos) / length(pos)

"""
    gyration_tensor(pos) -> SMatrix{3,3}

`S = (1/N) Σ_i (r_i - cm)(r_i - cm)ᵀ`.
"""
function gyration_tensor(pos::Vector{Vec3})
    cm = center_of_mass(pos)
    S = zero(SMatrix{3,3,Float64})
    @inbounds for r in pos
        d = r - cm
        S += d * d'
    end
    return S / length(pos)
end

"""
    radius_of_gyration(pos) -> Float64

`R_G = sqrt(tr S)`. Primary observable.
"""
@inline function radius_of_gyration(pos::Vector{Vec3})
    cm = center_of_mass(pos)
    s2 = 0.0
    @inbounds for r in pos
        d = r - cm
        s2 += dot(d, d)
    end
    return sqrt(s2 / length(pos))
end

"""
    radius_of_gyration_sq(pos) -> Float64

`R_G² = tr S`, without the square root.
"""
@inline function radius_of_gyration_sq(pos::Vector{Vec3})
    cm = center_of_mass(pos)
    s2 = 0.0
    @inbounds for r in pos
        d = r - cm
        s2 += dot(d, d)
    end
    return s2 / length(pos)
end

"""
    gyration_eigenvalues(pos) -> (λ1, λ2, λ3)  with λ1 ≥ λ2 ≥ λ3 ≥ 0
"""
function gyration_eigenvalues(pos::Vector{Vec3})
    S = gyration_tensor(pos)
    λ = eigvals(Symmetric(Matrix(S)))
    return (λ[3], λ[2], λ[1])
end

"""
    end_to_end(pos) -> Vec3
"""
@inline end_to_end(pos::Vector{Vec3}) = pos[end] - pos[1]

"""
    end_to_end_sq(pos) -> Float64

`R_e² = |r_N - r_1|²`, without the square root. Logged during production.
"""
@inline function end_to_end_sq(pos::Vector{Vec3})
    d = end_to_end(pos)
    return dot(d, d)
end

"""
    backbone_cosangles(pos) -> Vector{Float64}

`cosθ_i = û_{i-1,i} · û_{i,i+1}` at interior monomers i = 2 … N-1.
"""
function backbone_cosangles(pos::Vector{Vec3})
    N = length(pos)
    out = Vector{Float64}(undef, max(N - 2, 0))
    @inbounds for i in 2:N-1
        u1 = pos[i] - pos[i-1]
        u2 = pos[i+1] - pos[i]
        out[i-1] = dot(u1, u2) / (norm(u1) * norm(u2))
    end
    return out
end

"""
    AsphericityAccumulator()

Accumulates ⟨Tr²⟩ and ⟨3M⟩ separately over frames so the asphericity ratio
`A = ⟨Tr² - 3M⟩ / ⟨Tr²⟩` is formed from ensemble averages, never per-frame
ratios (§14). `Tr = λ1+λ2+λ3`, `M = λ1λ2 + λ1λ3 + λ2λ3`.
"""
mutable struct AsphericityAccumulator
    sum_tr2::Float64
    sum_3m::Float64
    n::Int
end
AsphericityAccumulator() = AsphericityAccumulator(0.0, 0.0, 0)

function accumulate!(acc::AsphericityAccumulator, pos::Vector{Vec3})
    λ1, λ2, λ3 = gyration_eigenvalues(pos)
    tr = λ1 + λ2 + λ3
    M = λ1 * λ2 + λ1 * λ3 + λ2 * λ3
    acc.sum_tr2 += tr^2
    acc.sum_3m += 3 * M
    acc.n += 1
    return acc
end

"""
    asphericity(acc) -> Float64

`A = ⟨Tr² - 3M⟩ / ⟨Tr²⟩ = (⟨Tr²⟩ - ⟨3M⟩)/⟨Tr²⟩`. A = 0 for a sphere,
A = 1 for a rod.
"""
function asphericity(acc::AsphericityAccumulator)
    acc.n == 0 && return NaN
    return (acc.sum_tr2 - acc.sum_3m) / acc.sum_tr2
end

# --- Potential energies --------------------------------------------------
#
# These mirror the loops in forces.jl exactly — same bond list, same
# `bonded_exclusion`, same strict `r < cutoff` test — accumulating `energy`
# instead of calling `pair_force`. That duplication is deliberate (the force
# loop stays allocation-free and branch-free) but it must not drift: the
# finite-difference tests in test/test_energy.jl differentiate these sums and
# compare against the force loops, and will fail if the two ever disagree.
#
# The active force is non-conservative (§2.3) and therefore has no energy; it
# never appears here.

"""
    bonded_energy(pos, bonded) -> Float64
    bonded_energy(sys) -> Float64

Total backbone spring energy `Σ_i V(|r_i - r_{i+1}|)` over the N-1 bonds.
"""
function bonded_energy(pos::Vector{Vec3}, bonded::BondedPotential)
    N = length(pos)
    U = 0.0
    @inbounds for i in 1:N-1
        U += energy(bonded, norm(pos[i] - pos[i+1]))
    end
    return U
end

bonded_energy(sys::System) = bonded_energy(sys.positions, sys.bonded)

"""
    pair_energy(pos, pair, neighbors) -> Float64

Total non-bonded energy: `Σ_{i<j} V(r_ij)` over pairs that are not backbone
neighbors and lie inside the cutoff. Selection logic is identical to
[`add_pair_forces!`](@ref).
"""
function pair_energy(pos::Vector{Vec3}, pair::PairPotential, ::AllPairs)
    N = length(pos)
    rc = cutoff(pair)
    rc2 = rc * rc
    U = 0.0
    @inbounds for i in 1:N-1
        for j in i+1:N
            bonded_exclusion(i, j) && continue
            d = pos[i] - pos[j]
            r2 = dot(d, d)
            r2 < rc2 || continue          # r >= cutoff contributes nothing
            U += energy(pair, sqrt(r2))
        end
    end
    return U
end

# Matches the VerletList fallback in forces.jl so energies and forces stay
# consistent under either strategy.
pair_energy(pos::Vector{Vec3}, pair::PairPotential, ::VerletList) =
    pair_energy(pos, pair, AllPairs())

"""
    nonbonded_energy(sys) -> Float64

Total non-bonded energy of `sys` at its current coupling.
"""
nonbonded_energy(sys::System) = pair_energy(sys.positions, sys.pair, sys.neighbors)

"""
    nonbonded_energies(sys) -> (U_nb, dUdc)

Both non-bonded energy quantities from a **single** pair loop: the total energy
`U_nb` and its derivative with respect to the coupling parameter,
`dUdc = ∂U_nb/∂c`, the observable thermodynamic integration averages.

The loop is run at unit coupling, giving `dUdc` directly; `U_nb = c * dUdc`
then follows exactly from the linearity invariant documented in
`potentials_pair.jl`. Doing it this way (rather than summing at `c` and
dividing) keeps `dUdc` well defined at `c = 0`, where `U_nb` vanishes and the
quotient would be 0/0 — that endpoint is the phantom chain, the whole reason
the observable exists.
"""
function nonbonded_energies(sys::System)
    dUdc = pair_energy(sys.positions, with_coupling(sys.pair, 1.0), sys.neighbors)
    return (coupling(sys.pair) * dUdc, dUdc)
end
