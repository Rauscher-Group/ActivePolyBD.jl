#!/usr/bin/env julia
# Micro-benchmark: steps/s for an N=40 active chain, and an allocation check
# on the hot path.  julia scripts/bench.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ActivePolyBD
using Random
using Printf

function bench(N=40, nsteps=200_000; Pe=10.0)
    rng = Xoshiro(1)
    pos = initialize_chain(N; mode=:line, rng=rng)
    sys = System(pos; bonded=HarmonicBond(100.0, 1.0),
                 pair=SoftRepulsive(100.0, 1.0),
                 activity=TangentialActivity(Pe), neighbors=AllPairs())
    dt = 1e-3

    bd_step!(sys, dt, rng)                          # warm up / compile
    alloc = @allocated bd_step!(sys, dt, rng)

    t0 = time()
    for _ in 1:nsteps
        bd_step!(sys, dt, rng)
    end
    elapsed = time() - t0
    @printf("N=%d  %d steps in %.3f s  =>  %.3g steps/s\n", N, nsteps, elapsed, nsteps / elapsed)
    @printf("allocated per bd_step!: %d bytes\n", alloc)
    return nsteps / elapsed
end

bench()
