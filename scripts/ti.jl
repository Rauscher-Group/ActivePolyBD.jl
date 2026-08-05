#!/usr/bin/env julia
# Thermodynamic integration from the phantom chain (non-bonded k = 0) to the
# full passive model, then integrate ⟨∂U/∂k⟩ over k to get ΔF:
#
#   julia -t auto scripts/ti.jl scripts/ti.toml
#
# Each k value gets its own subdirectory <out_dir>/k_<value>/ holding the
# per-replica logs. After the runs finish, analysis/thermo_integration.jl reads
# the dUdc column from them and reports ΔF.
#
# Only interactions.nonbonded.k varies; the bonded potential is held fixed.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ActivePolyBD
using TOML
using Printf

# Included at top level: including inside `main` and then touching the module
# binding in the same world age trips Julia 1.12's stricter world-age rules.
include(joinpath(@__DIR__, "..", "analysis", "thermo_integration.jl"))

k_dirname(k) = "k_" * replace(@sprintf("%g", k), "." => "p", "-" => "m")

"""
    ti_grid(ti) -> Vector{Float64}

The coupling values to simulate: either the explicit `k_values` list, or a grid
uniform in `s = ln(1 + k/a)` spanning `[k_min, k_max]`. The substituted spacing
clusters points where ⟨∂U/∂k⟩ actually varies (small k) and thins them out in
the C/k tail, which is where a uniform-in-k grid wastes most of its budget.
Prefer an odd `n_points` so the interval count is even and composite Simpson
applies throughout.
"""
function ti_grid(ti::AbstractDict)
    if haskey(ti, "k_values")
        ks = Float64.(ti["k_values"])
        length(ks) >= 2 || error("ti.k_values needs at least 2 entries")
        issorted(ks) || error("ti.k_values must be sorted ascending; got $ks")
        return ks
    end
    a = Float64(get(ti, "a", 1.0))
    kmin = Float64(get(ti, "k_min", 0.0))
    kmax = Float64(get(ti, "k_max", 100.0))
    n = Int(get(ti, "n_points", 13))
    (a > 0 && kmax > kmin >= 0 && n >= 3) ||
        error("ti grid needs a > 0, k_max > k_min ≥ 0, n_points ≥ 3")
    iseven(n) && @warn "ti.n_points is even; the last Simpson panel falls back to \
                        trapezoid. An odd count integrates more accurately." n_points = n
    smin, smax = log(1 + kmin / a), log(1 + kmax / a)
    ks = [a * (exp(smin + (smax - smin) * (i - 1) / (n - 1)) - 1) for i in 1:n]
    ks[1], ks[end] = kmin, kmax    # snap: the round trip through log/exp drifts
    return ks
end

"""
    check_config!(base)

Reject configurations for which the integration is not defined, before burning
any CPU on them.
"""
function check_config!(base::AbstractDict)
    # The tangential active force is non-conservative (§2.3): it is not the
    # gradient of any potential, so there is no free energy to integrate. TI is
    # only meaningful for the passive system.
    act = get(base, "activity", Dict{String,Any}("type" => "none"))
    atype = String(get(act, "type", "none"))
    Pe = Float64(get(act, "Pe", 0.0))
    (atype == "none" || Pe == 0.0) ||
        error("thermodynamic integration requires Pe = 0: the active force is \
               non-conservative and has no associated free energy (got \
               activity.type = \"$atype\", Pe = $Pe)")

    # Linear-coupling TI down to k = 0 needs a potential that stays bounded as
    # r → 0. SoftRepulsive is a finite parabola; WCA diverges as r^-12 and its
    # ε → 0 endpoint is the classic soft-core singularity.
    ptype = String(base["interactions"]["nonbonded"]["type"])
    ptype == "soft_repulsive" ||
        error("thermodynamic integration in this script requires \
               interactions.nonbonded.type = \"soft_repulsive\" (finite at r = 0); \
               got \"$ptype\". Linear coupling to zero is ill-behaved for a \
               diverging potential such as WCA.")
    return nothing
end

function main(args)
    if length(args) != 1
        println(stderr, "usage: julia scripts/ti.jl <ti.toml>")
        return 1
    end
    base = TOML.parsefile(args[1])
    ti = base["ti"]
    check_config!(base)

    a = Float64(get(ti, "a", 1.0))
    seed_offset = Bool(get(ti, "seed_offset", true))
    out_dir = String(get(ti, "out_dir", "ti_out"))
    k_values = ti_grid(ti)
    mkpath(out_dir)

    @info "Thermodynamic integration" n_points=length(k_values) a=a out_dir=out_dir
    @info "  k grid" k=join((@sprintf("%g", k) for k in k_values), " ")

    base_seed = UInt64(get(get(base, "system", Dict{String,Any}()), "seed", 12345))

    for (idx, k) in enumerate(k_values)
        subdir = joinpath(out_dir, k_dirname(k))
        mkpath(subdir)
        write(joinpath(subdir, "k.txt"), string(k))   # label for the analysis
        # Clone the base config, set this coupling and route outputs into subdir.
        cfg_dict = deepcopy(base)
        delete!(cfg_dict, "ti")
        cfg_dict["interactions"]["nonbonded"]["k"] = k
        # Offset the seed per point so the k-points are statistically
        # independent; sharing one seed correlates their errors and would make
        # the propagated error bar on ΔF too small.
        seed_offset && (cfg_dict["system"]["seed"] = base_seed + UInt64(idx))
        out = get!(cfg_dict, "output", Dict{String,Any}())
        out["xyz_file"] = joinpath(subdir, "traj.xyz")
        out["log_file"] = joinpath(subdir, "obs.dat")
        out["checkpoint_file"] = joinpath(subdir, "checkpoint.jls")

        cfg = parse_config(cfg_dict)
        @info "TI point" idx=idx of=length(k_values) k=k dir=subdir n_replicas=cfg.n_replicas
        t0 = time()
        run!(cfg)
        @info "  finished" k=k elapsed_s=round(time() - t0; digits=1)
    end

    @info "All TI points done; integrating"
    ThermoIntegration.analyze(out_dir; a=a)
    return 0
end

exit(main(ARGS))
