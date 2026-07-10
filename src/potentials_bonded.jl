# Bonded (backbone) potentials.
#
# Each concrete `BondedPotential` supplies:
#   dVdr(p, r)   -> V'(r), the radial derivative of the pair energy
#   energy(p, r) -> V(r)  (optional, used by tests / diagnostics)
#
# The force loop (forces.jl) turns V'(r) into a vector force via the shared
# geometry rule (§2.6), so a new bond type never touches the integrator.

abstract type BondedPotential end

"""
    HarmonicBond(k, r0)

Two-sided harmonic spring `V(r) = (k/2)(r - r0)^2`. Restores in both
directions. This is the paper's backbone bond with `k = K_sp = 100`,
`r0 = b = 1` in reduced units.
"""
struct HarmonicBond <: BondedPotential
    k::Float64
    r0::Float64
end
@inline dVdr(p::HarmonicBond, r) = p.k * (r - p.r0)
@inline energy(p::HarmonicBond, r) = 0.5 * p.k * (r - p.r0)^2

"""
    FENEBond(k, R0)

Finitely-extensible nonlinear elastic bond (Kremer–Grest form of the
attractive part), `V(r) = -(k/2) R0^2 log(1 - (r/R0)^2)`. Provided to
demonstrate the extension recipe (req. 7): a new struct + two methods,
no changes to the force loop or integrator. `R0` is the maximum extension.
"""
struct FENEBond <: BondedPotential
    k::Float64
    R0::Float64
end
@inline dVdr(p::FENEBond, r) = p.k * r / (1 - (r / p.R0)^2)
@inline energy(p::FENEBond, r) = -0.5 * p.k * p.R0^2 * log(1 - (r / p.R0)^2)
