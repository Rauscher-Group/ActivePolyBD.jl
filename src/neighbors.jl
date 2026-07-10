# Neighbor strategies for the non-bonded pair loop.
#
# A strategy decides which (i, j) pairs the pair loop visits. Backbone-bonded
# neighbors (|i-j| == 1) are always excluded via `bonded_exclusion`, so the
# exclusion rule lives here and not inside any potential.

abstract type NeighborStrategy end

"""
    bonded_exclusion(i, j)

True when monomers `i` and `j` are backbone neighbors and must be skipped by
the non-bonded pair loop (their interaction is handled by the bond).
"""
@inline bonded_exclusion(i::Int, j::Int) = abs(i - j) == 1

"""
    AllPairs()

Brute-force `i < j` double loop. For N ≤ 40 (~N²/2 ≈ 800 pairs) this beats
any neighbor list and allocates nothing. Default strategy.
"""
struct AllPairs <: NeighborStrategy end

"""
    VerletList(skin)

Interface-complete stub for future large-N runs: a Verlet list rebuilt when
the max monomer displacement since the last build exceeds `skin/2`. Not used
for the α = 0, N ≤ 40 scope; `AllPairs` is faster there. Left here so the
force loop can dispatch on it unchanged once the rebuild logic is filled in.
"""
struct VerletList <: NeighborStrategy
    skin::Float64
end
