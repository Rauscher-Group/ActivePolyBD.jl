# Force accumulation: bonded + non-bonded + active.
#
# All pair/bond geometry lives in one place (§2.6). Given d = r_i - r_j,
# r = |d|, and fmag = V'(r):
#     f_i = -fmag * d/r ,   f_j = +fmag * d/r
# so a stretched harmonic bond (V' > 0) pulls i toward j. Each potential only
# supplies V'(r); no allocations inside the loops.

@inline function pair_force(pot, d::Vec3, r::Float64)
    fmag = dVdr(pot, r)
    return (-fmag / r) * d
end

"""
    add_bonded_forces!(forces, pos, bonded)

Backbone springs between consecutive monomers i–(i+1).
"""
function add_bonded_forces!(forces::Vector{Vec3}, pos::Vector{Vec3}, bonded::BondedPotential)
    N = length(pos)
    @inbounds for i in 1:N-1
        d = pos[i] - pos[i+1]
        r = norm(d)
        f = pair_force(bonded, d, r)
        forces[i] += f
        forces[i+1] -= f
    end
    return forces
end

"""
    add_pair_forces!(forces, pos, pair, neighbors)

Non-bonded pair interactions. `AllPairs`: brute-force i<j, skipping
backbone-bonded neighbors and any pair beyond the potential cutoff.
"""
function add_pair_forces!(forces::Vector{Vec3}, pos::Vector{Vec3},
                          pair::PairPotential, ::AllPairs)
    N = length(pos)
    rc = cutoff(pair)
    rc2 = rc * rc
    @inbounds for i in 1:N-1
        for j in i+1:N
            bonded_exclusion(i, j) && continue
            d = pos[i] - pos[j]
            r2 = dot(d, d)
            r2 < rc2 || continue          # r >= cutoff contributes nothing
            r = sqrt(r2)
            f = pair_force(pair, d, r)
            forces[i] += f
            forces[j] -= f
        end
    end
    return forces
end

# VerletList falls back to AllPairs semantics until the rebuild logic lands
# (Phase 7). Keeping identical results lets it be swapped in transparently.
function add_pair_forces!(forces::Vector{Vec3}, pos::Vector{Vec3},
                          pair::PairPotential, ::VerletList)
    return add_pair_forces!(forces, pos, pair, AllPairs())
end

"""
    compute_forces!(sys)

Zero the accumulator, then add bonded, non-bonded, and active contributions.
"""
function compute_forces!(sys::System)
    fill!(sys.forces, zero(Vec3))
    add_bonded_forces!(sys.forces, sys.positions, sys.bonded)
    add_pair_forces!(sys.forces, sys.positions, sys.pair, sys.neighbors)
    add_active_forces!(sys.forces, sys.positions, sys.activity)
    return sys.forces
end
