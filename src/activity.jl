# Active-force models.
#
# An `ActivityModel` fills its contribution into the force accumulator. The
# active force is NON-CONSERVATIVE (§2.3): it is not the gradient of any
# potential, so it is added directly to `forces` and must never enter an
# energy. Potential energy is therefore not a valid stability check.

abstract type ActivityModel end

"""
    NoActivity()

Passive chain: no active force. Terminal monomers and interior monomers
alike feel nothing beyond the conservative forces.
"""
struct NoActivity <: ActivityModel end

"""
    TangentialActivity(Pe)

α = 0 tangential self-propulsion (§2.3). For interior monomers
`i = 2 … N-1` (1-based) the propulsion direction is the normalized
"skip-one" tangent `t_i = r_{i+1} - r_{i-1}`:

    f_i^act = Pe * t_i / |t_i|

The first and last monomers (i = 1, i = N) are PASSIVE. In reduced units
the active-force magnitude equals the Péclet number, `f^act = Pe` (§2.5).
"""
struct TangentialActivity <: ActivityModel
    Pe::Float64
end

# Passive: nothing to add.
add_active_forces!(forces, pos, ::NoActivity) = forces

function add_active_forces!(forces, pos, a::TangentialActivity)
    N = length(pos)
    @inbounds for i in 2:N-1
        t = pos[i+1] - pos[i-1]          # skip-one tangent, NOT an adjacent bond
        forces[i] += a.Pe * (t / norm(t))
    end
    return forces
end

# --- Future extension point (do NOT implement now) -----------------------
# A `ConeActivity{α}` for α > 0 would carry per-monomer orientation state
# (unit vectors) evolved by rotational diffusion and reflected into a cone
# of half-angle α about the local tangent (SM Eq. 2). It would slot in here
# as another `ActivityModel` with its own `add_active_forces!` plus a state
# update hook called from the integrator; nothing else would change.
