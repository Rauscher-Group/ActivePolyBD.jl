# Core data types.
#
# `System` is concretely parameterized on its potential/activity/neighbor
# types so every force call specializes and inlines — type stability is the
# single biggest performance lever here (§5.1).

const Vec3 = SVector{3,Float64}

"""
    System(N, positions, forces, bonded, pair, activity, neighbors)

A bead–spring chain of `N` monomers living in unbounded ℝ³ (no box).
`positions` and `forces` are length-`N` vectors of `SVector{3,Float64}`;
`forces` is preallocated scratch overwritten each step.
"""
struct System{B<:BondedPotential,P<:PairPotential,A<:ActivityModel,NS<:NeighborStrategy}
    N::Int
    positions::Vector{Vec3}
    forces::Vector{Vec3}
    bonded::B
    pair::P
    activity::A
    neighbors::NS
end

"""
    System(positions; bonded, pair, activity, neighbors=AllPairs())

Convenience constructor allocating the force scratch buffer.
"""
function System(positions::Vector{Vec3};
                bonded::BondedPotential,
                pair::PairPotential,
                activity::ActivityModel,
                neighbors::NeighborStrategy=AllPairs())
    N = length(positions)
    forces = zeros(Vec3, N)
    return System(N, positions, forces, bonded, pair, activity, neighbors)
end

"""
    SimParams(; dt, n_equil, n_prod, xyz_every, log_every, checkpoint_every, seed)

Run-control parameters. All frequencies are in steps; `dt` is in units of
τ_B (reduced). `seed` seeds the base RNG (each replica derives its own).
"""
struct SimParams
    dt::Float64
    n_equil::Int
    n_prod::Int
    xyz_every::Int
    log_every::Int
    checkpoint_every::Int
    seed::UInt64
end

function SimParams(; dt::Real=1e-3,
                   n_equil::Integer=1_000_000,
                   n_prod::Integer=10_000_000,
                   xyz_every::Integer=1000,
                   log_every::Integer=1000,
                   checkpoint_every::Integer=1_000_000,
                   seed::Integer=12345)
    return SimParams(Float64(dt), Int(n_equil), Int(n_prod), Int(xyz_every),
                     Int(log_every), Int(checkpoint_every), UInt64(seed))
end

"""
    deepcopy_system(sys)

An independent copy of `sys` with freshly allocated position/force buffers
(potentials/activity/neighbors are immutable and shared). Used to give each
replica its own state.
"""
function deepcopy_system(sys::System)
    return System(sys.N, copy(sys.positions), copy(sys.forces),
                  sys.bonded, sys.pair, sys.activity, sys.neighbors)
end
