# Config parsing/validation, XYZ/log writing, and checkpoint round-trip (§7,§8).

using TOML

@testset "config parsing and registries" begin
    cfg = parse_config(TOML.parse("""
        [system]
        N = 40
        seed = 7
        init = "line"
        [interactions.bonded]
        type = "harmonic"
        k = 100.0
        r0 = 1.0
        [interactions.nonbonded]
        type = "soft_repulsive"
        k = 100.0
        cutoff = 1.0
        [activity]
        type = "tangential"
        Pe = 12.0
        [integrator]
        dt = 1e-3
        equilibration_steps = 10
        production_steps = 20
        [replicas]
        n_replicas = 2
        parallel = "serial"
    """))
    @test cfg.N == 40
    @test cfg.bonded isa HarmonicBond
    @test cfg.pair isa SoftRepulsive
    @test cfg.activity isa TangentialActivity && cfg.activity.Pe == 12.0
    @test cfg.n_replicas == 2 && cfg.parallel === :serial

    # FENE / WCA reachable purely through the registry (req. 7).
    cfg2 = parse_config(TOML.parse("""
        [system]
        N = 5
        [interactions.bonded]
        type = "fene"
        k = 30.0
        R0 = 1.5
        [interactions.nonbonded]
        type = "wca"
        eps = 1.0
        sigma = 1.0
        [activity]
        type = "none"
        [integrator]
        dt = 1e-3
    """))
    @test cfg2.bonded isa FENEBond
    @test cfg2.pair isa WCA
    @test cfg2.activity isa NoActivity
end

@testset "config validation errors" begin
    bad(toml) = parse_config(TOML.parse(toml))
    base = """
        [interactions.bonded]
        type="harmonic"
        k=1.0
        r0=1.0
        [interactions.nonbonded]
        type="soft_repulsive"
        k=1.0
        cutoff=1.0
        [integrator]
        dt=1e-3
    """
    @test_throws ArgumentError bad("[system]\nN=2\n" * base)              # N < 3
    @test_throws ArgumentError bad("[system]\nN=5\n[activity]\ntype=\"tangential\"\nPe=-1.0\n" * base)  # Pe < 0
    @test_throws ArgumentError bad("[system]\nN=5\n" * replace(base, "dt=1e-3" => "dt=-1.0"))          # dt <= 0
    @test_throws ArgumentError bad("[system]\nN=5\n[interactions.bonded]\ntype=\"nope\"\nk=1.0\nr0=1.0\n[interactions.nonbonded]\ntype=\"soft_repulsive\"\nk=1.0\ncutoff=1.0\n[integrator]\ndt=1e-3\n")  # unknown type
end

@testset "XYZ and log writing" begin
    rng = Xoshiro(5)
    pos = initialize_chain(6; mode=:line, rng=rng)
    sys = System(pos; bonded=HarmonicBond(100.0, 1.0),
                 pair=SoftRepulsive(100.0, 1.0), activity=TangentialActivity(3.0))

    # :line init already starts bead 1 at the origin, so displace the chain
    # first — otherwise the recentering test would pass vacuously.
    offset = ActivePolyBD.Vec3(4.0, -2.5, 1.5)
    sys.positions .= [r + offset for r in sys.positions]
    before = copy(sys.positions)

    mktemp() do path, io
        write_xyz_frame!(io, sys, 100, 0.1, radius_of_gyration(sys.positions))
        flush(io)
        lines = readlines(path)
        @test parse(Int, lines[1]) == 6
        @test occursin("step=100", lines[2])
        @test startswith(lines[3], "O")            # first bead: passive end
        @test startswith(lines[4], "C")            # interior
        @test startswith(lines[8], "O")            # last bead: passive end

        coords(l) = parse.(Float64, split(l)[2:4])
        @test coords(lines[3]) == [0.0, 0.0, 0.0]  # bead 1 placed at the origin
        # A shift, not a distortion: separations survive and the last bead
        # lands exactly at r_N − r_1.
        @test isapprox(norm(coords(lines[4]) .- coords(lines[3])),
                       norm(sys.positions[2] - sys.positions[1]); rtol=1e-6)
        @test isapprox(coords(lines[8]),
                       collect(sys.positions[6] - sys.positions[1]); rtol=1e-6)
        @test sys.positions == before              # write-time shift only
    end

    mktemp() do path, io
        write_log_header(io)
        write_log_row!(io, 100, 0.1, 1.23, center_of_mass(sys.positions),
                       end_to_end_sq(sys.positions))
        flush(io)
        lines = readlines(path)
        @test startswith(lines[1], "#")
        @test split(lines[1])[end] == "Re2"        # R_e² appended last
        @test length(split(lines[2])) == 7
        @test isapprox(parse(Float64, split(lines[2])[7]),
                       end_to_end_sq(sys.positions); rtol=1e-6)
    end
end

@testset "checkpoint round-trip and restart" begin
    rng = Xoshiro(8)
    pos = initialize_chain(10; mode=:line, rng=rng)
    sys = System(pos; bonded=HarmonicBond(100.0, 1.0),
                 pair=SoftRepulsive(100.0, 1.0), activity=TangentialActivity(4.0))
    for _ in 1:500
        bd_step!(sys, 1e-3, rng)
    end

    path = tempname() * ".jls"
    save_checkpoint(path, sys, 500, rng)

    # Continue the "reference" run 500 more steps.
    ref = deepcopy_system(sys)
    refrng = deepcopy(rng)
    for _ in 1:500
        bd_step!(ref, 1e-3, refrng)
    end

    # Restart from checkpoint into a fresh system and continue.
    fresh = System(10, similar(pos), zeros(ActivePolyBD.Vec3, 10),
                   HarmonicBond(100.0, 1.0), SoftRepulsive(100.0, 1.0),
                   TangentialActivity(4.0), AllPairs())
    step, rrng = restart!(fresh, path)
    @test step == 500
    for _ in 1:500
        bd_step!(fresh, 1e-3, rrng)
    end
    @test fresh.positions == ref.positions        # restart reproduces continuation
    rm(path; force=true)
end
