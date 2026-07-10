# Euler–Maruyama Brownian-dynamics integrator (reduced units, §2.6).
#
#   r_i <- r_i + (F_i^cons + F_i^act) * dt + sqrt(2*dt) * ξ_i
#
# with ξ_i three independent N(0,1) draws per step per component. In reduced
# units D₀ = βD₀ = 1, so the deterministic prefactor on the force is dt and
# the noise amplitude is sqrt(2*dt). The off-by-√2 here silently rescales
# temperature (§14) — the amplitude is exactly sqrt(2*dt), not sqrt(dt).

"""
    bd_step!(sys, dt, rng)

Advance the whole chain one Euler–Maruyama step in place.
"""
function bd_step!(sys::System, dt::Float64, rng)
    compute_forces!(sys)
    s = sqrt(2 * dt)
    @inbounds for i in 1:sys.N
        ξ = Vec3(randn(rng), randn(rng), randn(rng))
        sys.positions[i] += sys.forces[i] * dt + s * ξ
    end
    return sys
end
