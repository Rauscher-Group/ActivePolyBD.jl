# Simulation driver and replicas (§9).
#
# Replicas are fully independent (own System, own RNG stream seeded
# seed ⊻ replica_id, own suffixed output files) and embarrassingly parallel.

"""
    ReplicaResult(replica_id, rg_samples, log_path, xyz_path)

Per-replica production output: the in-memory R_G time series (sampled at
`log_every`) plus the paths written.
"""
struct ReplicaResult
    replica_id::Int
    rg_samples::Vector{Float64}
    log_path::String
    xyz_path::String
end

"""
    replica_rng(seed, replica_id) -> Xoshiro

Independent, well-separated stream per replica (§14). Distinct ids give
distinct trajectories; the same (seed, id) reproduces bit-for-bit.
"""
replica_rng(seed::UInt64, replica_id::Integer) = Xoshiro(seed ⊻ (0x9e3779b97f4a7c15 * UInt64(replica_id)))

"""
    suffixed(path, id) -> String

Insert `_r{id}` before the extension: `traj.xyz` → `traj_r3.xyz`.
"""
function suffixed(path::AbstractString, id::Integer)
    base, ext = splitext(path)
    return string(base, "_r", id, ext)
end

"""
    run_single!(sys, params, rng, output; replica_id=0, restart_path=nothing,
                collect_rg=true) -> ReplicaResult

Equilibrate `n_equil` steps (no data written), then run `n_prod` production
steps, writing the XYZ trajectory, scalar log, and checkpoints at their
strides. The center of mass is never recentered during production — its free
drift is physical (§14). If `restart_path` points to a checkpoint, positions,
step, and RNG are loaded and production continues from there (equilibration
is skipped).
"""
function run_single!(sys::System, params::SimParams, rng, output::OutputSpec;
                     replica_id::Integer=0, restart_path=nothing, collect_rg::Bool=true)
    dt = params.dt
    start_step = 0

    if restart_path !== nothing && isfile(restart_path)
        start_step, rng = restart!(sys, restart_path)
    else
        for _ in 1:params.n_equil
            bd_step!(sys, dt, rng)
        end
    end

    xyz_path = suffixed(output.xyz_file, replica_id)
    log_path = suffixed(output.log_file, replica_id)
    ckpt_path = suffixed(output.checkpoint_file, replica_id)

    rg_samples = Float64[]
    xyz_io = open(xyz_path, start_step == 0 ? "w" : "a")
    log_io = open(log_path, start_step == 0 ? "w" : "a")
    try
        start_step == 0 && write_log_header(log_io)
        for k in (start_step + 1):params.n_prod
            bd_step!(sys, dt, rng)
            t = k * dt
            if params.log_every > 0 && k % params.log_every == 0
                Rg = radius_of_gyration(sys.positions)
                cm = center_of_mass(sys.positions)
                write_log_row!(log_io, k, t, Rg, cm)
                collect_rg && push!(rg_samples, Rg)
            end
            if params.xyz_every > 0 && k % params.xyz_every == 0
                Rg = radius_of_gyration(sys.positions)
                write_xyz_frame!(xyz_io, sys, k, t, Rg)
            end
            if params.checkpoint_every > 0 && k % params.checkpoint_every == 0
                flush(xyz_io); flush(log_io)
                save_checkpoint(ckpt_path, sys, k, rng)
            end
        end
    finally
        close(xyz_io)
        close(log_io)
    end

    return ReplicaResult(replica_id, rg_samples, log_path, xyz_path)
end

"""
    run!(cfg::RunConfig) -> Vector{ReplicaResult}

Build and run `n_replicas` independent replicas. Each gets its own freshly
initialized `System`, its own RNG stream, and output files suffixed by
replica index. `parallel = :threads` runs them with `Threads.@threads`
(start Julia with `-t auto`); `:serial` runs them in order (debugging).
"""
function run!(cfg::RunConfig)
    results = Vector{ReplicaResult}(undef, cfg.n_replicas)
    if cfg.parallel === :threads
        Threads.@threads for id in 1:cfg.n_replicas
            results[id] = _run_replica(cfg, id)
        end
    else
        for id in 1:cfg.n_replicas
            results[id] = _run_replica(cfg, id)
        end
    end
    return results
end

function _run_replica(cfg::RunConfig, id::Integer)
    rng = replica_rng(cfg.params.seed, id)
    sys = build_system(cfg, rng)
    return run_single!(sys, cfg.params, rng, cfg.output; replica_id=id)
end
