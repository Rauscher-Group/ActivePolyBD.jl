# Chain initialization (§5.8). Produce a starting configuration with bond
# lengths ≈ b and no gross overlaps, then rely on equilibration.

"""
    initialize_chain(N; mode=:line, rng=Random.default_rng(), b=1.0, kink=0.1)

Build a length-`N` chain of `SVector{3,Float64}` positions.

- `:line` — a straight backbone along x with small random transverse kinks
  of magnitude `kink`, giving bond lengths near `b`.
- `:saw`  — a self-avoiding growth walk: each new bead is placed a distance
  `b` from the previous one in a random direction, rejecting placements that
  overlap an existing bead (r < b) until a valid one is found.
"""
function initialize_chain(N::Integer; mode::Symbol=:line,
                          rng=Random.default_rng(), b::Real=1.0, kink::Real=0.1)
    N >= 3 || throw(ArgumentError("N must be ≥ 3 so interior monomers exist (got $N)"))
    if mode === :line
        return _init_line(N, rng, Float64(b), Float64(kink))
    elseif mode === :saw
        return _init_saw(N, rng, Float64(b))
    else
        throw(ArgumentError("unknown init mode $(mode); use :line or :saw"))
    end
end

function _init_line(N::Int, rng, b::Float64, kink::Float64)
    pos = Vector{Vec3}(undef, N)
    x = 0.0
    prev = Vec3(0.0, 0.0, 0.0)
    pos[1] = prev
    for i in 2:N
        # step ~b along x with a small transverse perturbation, then rescale
        # the bond to exactly b so the harmonic bond starts near its minimum.
        step = Vec3(b, kink * randn(rng), kink * randn(rng))
        step = b * step / norm(step)
        prev = prev + step
        pos[i] = prev
    end
    return pos
end

function _init_saw(N::Int, rng, b::Float64)
    pos = Vector{Vec3}(undef, N)
    pos[1] = Vec3(0.0, 0.0, 0.0)
    i = 2
    attempts_total = 0
    while i <= N
        dir = randn(rng, Vec3)
        dir = dir / norm(dir)
        cand = pos[i-1] + b * dir
        ok = true
        @inbounds for j in 1:i-1
            if norm(cand - pos[j]) < b
                ok = false
                break
            end
        end
        if ok
            pos[i] = cand
            i += 1
        end
        attempts_total += 1
        if attempts_total > 1000 * N
            error("SAW initialization failed to place monomer $i after many attempts; try mode=:line")
        end
    end
    return pos
end
