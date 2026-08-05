# Non-bonded (pair) potentials.
#
# Each concrete `PairPotential` supplies:
#   dVdr(p, r)   -> V'(r)
#   energy(p, r) -> V(r)   (optional)
#   cutoff(p)    -> interaction range; pairs with r >= cutoff contribute nothing
#
# For thermodynamic integration each also supplies:
#   coupling(p)         -> the scalar prefactor that scales the whole potential
#   with_coupling(p, c) -> the same potential with that prefactor replaced
#
# The pair loop skips backbone-bonded neighbors (|i-j| == 1); that exclusion
# lives in the neighbor strategy, not here.

abstract type PairPotential end

"""
    SoftRepulsive(k, cutoff)

The paper's one-sided harmonic core (§2.2). Same parabola and stiffness as
the bond, but keeps **only the repulsive half**: active for `r < cutoff`
and identically zero for `r >= cutoff`. At `r = cutoff` both `V` and
`V' = k(r - cutoff)` vanish, so the force turns on continuously (C¹).
`cutoff = b = 1`, `k = K_sp = 100` in reduced units.
"""
struct SoftRepulsive <: PairPotential
    k::Float64
    cutoff::Float64
end
@inline cutoff(p::SoftRepulsive) = p.cutoff
@inline dVdr(p::SoftRepulsive, r) = r < p.cutoff ? p.k * (r - p.cutoff) : 0.0
@inline energy(p::SoftRepulsive, r) = r < p.cutoff ? 0.5 * p.k * (r - p.cutoff)^2 : 0.0

"""
    WCA(eps, sigma)

Weeks–Chandler–Andersen purely-repulsive Lennard-Jones, cut and shifted at
the minimum `r_c = 2^(1/6) σ`. Provided as an extension example (req. 7).

    V(r) = 4ε[(σ/r)^12 - (σ/r)^6] + ε,   r < r_c
    V'(r) = -24ε/r [2(σ/r)^12 - (σ/r)^6]
"""
struct WCA <: PairPotential
    eps::Float64
    sigma::Float64
end
@inline cutoff(p::WCA) = 2.0^(1 / 6) * p.sigma
@inline function dVdr(p::WCA, r)
    if r < cutoff(p)
        sr6 = (p.sigma / r)^6
        sr12 = sr6 * sr6
        return -24.0 * p.eps / r * (2.0 * sr12 - sr6)
    else
        return 0.0
    end
end
@inline function energy(p::WCA, r)
    if r < cutoff(p)
        sr6 = (p.sigma / r)^6
        sr12 = sr6 * sr6
        return 4.0 * p.eps * (sr12 - sr6) + p.eps
    else
        return 0.0
    end
end

# --- Coupling parameter (thermodynamic integration) ----------------------
#
# `coupling(p)` names the parameter that scales the whole potential, and
# `with_coupling(p, c)` rebuilds `p` with that parameter set to `c`. Together
# they let a TI driver vary the interaction strength without hard-coding which
# field to touch (`k` here, `eps` there).
#
# Both obey the invariant the TI machinery relies on — `energy` is linear and
# homogeneous in the coupling:
#
#     energy(with_coupling(p, c), r) == c * energy(with_coupling(p, 1), r)
#
# so ∂V/∂c is exactly the unit-coupling energy, and `cutoff` is
# coupling-independent, so which pairs interact never changes with `c`.
#
# Linear-coupling TI to c = 0 additionally requires V to stay *bounded* as
# r → 0. That holds for `SoftRepulsive` (a finite parabola) but NOT for `WCA`
# (r^-12 divergence): the accessors are defined for both for interface
# completeness, but only `SoftRepulsive` is a valid TI endpoint.

"""
    coupling(p) -> Float64

The scalar prefactor scaling the whole pair potential: `k` for
[`SoftRepulsive`](@ref), `eps` for [`WCA`](@ref).
"""
@inline coupling(p::SoftRepulsive) = p.k
@inline coupling(p::WCA) = p.eps

"""
    with_coupling(p, c) -> PairPotential

`p` with its coupling parameter replaced by `c`, all other parameters (and
therefore `cutoff`) unchanged.
"""
@inline with_coupling(p::SoftRepulsive, c::Real) = SoftRepulsive(Float64(c), p.cutoff)
@inline with_coupling(p::WCA, c::Real) = WCA(Float64(c), p.sigma)
