#!/usr/bin/env julia
# Run one chain across several Pe values (the headline sweep) and then invoke
# the analysis to produce P(R_G) vs Pe:
#
#   julia -t auto scripts/sweep.jl scripts/sweep.toml
#
# Each Pe value gets its own subdirectory <out_dir>/pe_<value>/ containing the
# per-replica logs. After the runs finish, analysis/rg_distribution.jl reads
# them and writes the overlay data + summary table.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ActivePolyBD
using TOML
using Printf

pe_dirname(pe) = "pe_" * replace(@sprintf("%g", pe), "." => "p", "-" => "m")

function main(args)
    if length(args) != 1
        println(stderr, "usage: julia scripts/sweep.jl <sweep.toml>")
        return 1
    end
    base = TOML.parsefile(args[1])
    sweep = base["sweep"]
    pe_values = Float64.(sweep["pe_values"])
    out_dir = String(get(sweep, "out_dir", "sweep_out"))
    mkpath(out_dir)

    for pe in pe_values
        subdir = joinpath(out_dir, pe_dirname(pe))
        mkpath(subdir)
        write(joinpath(subdir, "pe.txt"), string(pe))   # label for the analysis
        # Clone the base config, set this Pe and route outputs into subdir.
        cfg_dict = deepcopy(base)
        delete!(cfg_dict, "sweep")
        cfg_dict["activity"]["type"] = "tangential"
        cfg_dict["activity"]["Pe"] = pe
        out = get!(cfg_dict, "output", Dict{String,Any}())
        out["xyz_file"] = joinpath(subdir, "traj.xyz")
        out["log_file"] = joinpath(subdir, "obs.dat")
        out["checkpoint_file"] = joinpath(subdir, "checkpoint.jls")

        cfg = parse_config(cfg_dict)
        @info "Sweep point" Pe=pe dir=subdir n_replicas=cfg.n_replicas
        t0 = time()
        run!(cfg)
        @info "  finished" Pe=pe elapsed_s=round(time() - t0; digits=1)
    end

    @info "All sweep points done; running analysis"
    include(joinpath(@__DIR__, "..", "analysis", "rg_distribution.jl"))
    Base.invokelatest(RgDistribution.analyze, out_dir)
    return 0
end

exit(main(ARGS))
