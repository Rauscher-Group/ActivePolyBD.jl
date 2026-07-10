# Configuration / scripting interface (§8).
#
# A TOML script is parsed into typed objects via registries mapping
# type-strings → builder closures, so adding a potential makes it usable from
# config with no parser edits (req. 7). The Julia API (build System/SimParams
# directly) constructs the same objects.

# --- Registries ----------------------------------------------------------
# Each builder takes the relevant TOML subtable (a Dict) and returns a struct.

const BONDED_REGISTRY = Dict{String,Function}(
    "harmonic" => d -> HarmonicBond(Float64(d["k"]), Float64(d["r0"])),
    "fene"     => d -> FENEBond(Float64(d["k"]), Float64(d["R0"])),
)

const PAIR_REGISTRY = Dict{String,Function}(
    "soft_repulsive" => d -> SoftRepulsive(Float64(d["k"]), Float64(d["cutoff"])),
    "wca"            => d -> WCA(Float64(d["eps"]), Float64(d["sigma"])),
)

const ACTIVITY_REGISTRY = Dict{String,Function}(
    "tangential" => d -> TangentialActivity(Float64(get(d, "Pe", 0.0))),
    "none"       => d -> NoActivity(),
)

# --- Output specification ------------------------------------------------

struct OutputSpec
    xyz_file::String
    log_file::String
    checkpoint_file::String
end

# --- Parsed configuration ------------------------------------------------

"""
    RunConfig

Everything needed to launch a run: the (immutable, shareable) interaction
models, run parameters, output file bases, and replica settings. Per-replica
`System`s are built lazily by the driver so each gets an independent initial
configuration.
"""
struct RunConfig{B<:BondedPotential,P<:PairPotential,A<:ActivityModel}
    N::Int
    init_mode::Symbol
    bonded::B
    pair::P
    activity::A
    params::SimParams
    output::OutputSpec
    n_replicas::Int
    parallel::Symbol
end

"""
    build_system(cfg, rng) -> System

Construct a fresh `System` for one replica, initializing the chain with `rng`.
"""
function build_system(cfg::RunConfig, rng)
    pos = initialize_chain(cfg.N; mode=cfg.init_mode, rng=rng)
    return System(pos; bonded=cfg.bonded, pair=cfg.pair,
                  activity=cfg.activity, neighbors=AllPairs())
end

# --- TOML → RunConfig ----------------------------------------------------

_registry_lookup(reg, key, table) = haskey(reg, key) ? reg[key](table) :
    throw(ArgumentError("unknown type \"$key\"; known: $(sort(collect(keys(reg))))"))

"""
    load_config(path) -> RunConfig

Parse a TOML script into a validated `RunConfig`.
"""
function load_config(path::AbstractString)
    return parse_config(TOML.parsefile(path))
end

"""
    parse_config(dict) -> RunConfig

Build a `RunConfig` from an already-parsed TOML dictionary. Validates
positive dt, N ≥ 3, known type strings, and non-negative Pe with clear errors.
"""
function parse_config(cfg::AbstractDict)
    sysc = cfg["system"]
    N = Int(sysc["N"])
    N >= 3 || throw(ArgumentError("system.N must be ≥ 3 (interior monomers required); got $N"))
    dims = Int(get(sysc, "dimensions", 3))
    dims == 3 || throw(ArgumentError("only dimensions = 3 is supported in v1; got $dims"))
    init_mode = Symbol(get(sysc, "init", "line"))
    init_mode in (:line, :saw) || throw(ArgumentError("system.init must be \"line\" or \"saw\"; got $init_mode"))
    seed = UInt64(get(sysc, "seed", 12345))

    inter = cfg["interactions"]
    bonded = _registry_lookup(BONDED_REGISTRY, String(inter["bonded"]["type"]), inter["bonded"])
    pair = _registry_lookup(PAIR_REGISTRY, String(inter["nonbonded"]["type"]), inter["nonbonded"])

    act_table = get(cfg, "activity", Dict("type" => "none"))
    activity = _registry_lookup(ACTIVITY_REGISTRY, String(act_table["type"]), act_table)
    if activity isa TangentialActivity && activity.Pe < 0
        throw(ArgumentError("activity.Pe must be ≥ 0; got $(activity.Pe)"))
    end

    integ = cfg["integrator"]
    dt = Float64(integ["dt"])
    dt > 0 || throw(ArgumentError("integrator.dt must be > 0; got $dt"))
    n_equil = Int(get(integ, "equilibration_steps", 0))
    n_prod = Int(get(integ, "production_steps", 0))
    (n_equil >= 0 && n_prod >= 0) || throw(ArgumentError("step counts must be ≥ 0"))

    out = get(cfg, "output", Dict{String,Any}())
    xyz_every = Int(get(out, "xyz_every", 1000))
    log_every = Int(get(out, "log_every", 1000))
    checkpoint_every = Int(get(out, "checkpoint_every", 1_000_000))
    output = OutputSpec(String(get(out, "xyz_file", "traj.xyz")),
                        String(get(out, "log_file", "obs.dat")),
                        String(get(out, "checkpoint_file", "checkpoint.jls")))

    params = SimParams(; dt=dt, n_equil=n_equil, n_prod=n_prod,
                       xyz_every=xyz_every, log_every=log_every,
                       checkpoint_every=checkpoint_every, seed=seed)

    rep = get(cfg, "replicas", Dict{String,Any}())
    n_replicas = Int(get(rep, "n_replicas", 1))
    n_replicas >= 1 || throw(ArgumentError("replicas.n_replicas must be ≥ 1; got $n_replicas"))
    parallel = Symbol(get(rep, "parallel", "serial"))
    parallel in (:serial, :threads) ||
        throw(ArgumentError("replicas.parallel must be \"serial\" or \"threads\" (gpu is future); got $parallel"))

    return RunConfig(N, init_mode, bonded, pair, activity, params, output, n_replicas, parallel)
end
