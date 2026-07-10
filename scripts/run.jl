#!/usr/bin/env julia
# Run a single configuration:  julia -t auto scripts/run.jl config.toml
#
# Activates the package environment, parses the TOML script, and launches all
# replicas. Outputs are written to the files named in the [output] table,
# suffixed with the replica index.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ActivePolyBD
using Statistics

function main(args)
    if length(args) != 1
        println(stderr, "usage: julia scripts/run.jl <config.toml>")
        return 1
    end
    cfg = load_config(args[1])
    @info "Loaded config" N=cfg.N activity=typeof(cfg.activity) n_replicas=cfg.n_replicas parallel=cfg.parallel threads=Threads.nthreads()

    t0 = time()
    results = run!(cfg)
    elapsed = time() - t0

    total_steps = cfg.n_replicas * (cfg.params.n_equil + cfg.params.n_prod)
    all_rg = reduce(vcat, (r.rg_samples for r in results); init=Float64[])
    @info "Done" elapsed_s=round(elapsed; digits=2) steps_per_s=round(total_steps/elapsed; digits=0) n_rg_samples=length(all_rg)
    if !isempty(all_rg)
        @info "R_G (pooled over replicas)" mean=round(mean(all_rg); digits=4) std=round(std(all_rg); digits=4)
    end
    return 0
end

exit(main(ARGS))
