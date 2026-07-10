# Tests 2, 3, 5, 6 (§10): noise amplitude, passive CM diffusion, determinism,
# and dt-convergence of ⟨R_G⟩.

using Statistics

# Build a free single particle (N=1: no bonds, no pairs, no activity fire).
function free_particle()
    pos = [ActivePolyBD.Vec3(0.0, 0.0, 0.0)]
    return System(1, pos, zeros(ActivePolyBD.Vec3, 1),
                  HarmonicBond(100.0, 1.0), SoftRepulsive(100.0, 1.0),
                  NoActivity(), AllPairs())
end

@testset "free single particle MSD = 6t" begin
    dt = 0.01
    nsteps = 200
    M = 4000                     # independent walkers
    t = nsteps * dt
    msd = 0.0
    for m in 1:M
        sys = free_particle()
        rng = Xoshiro(UInt64(m))
        r0 = sys.positions[1]
        for _ in 1:nsteps
            bd_step!(sys, dt, rng)
        end
        d = sys.positions[1] - r0
        msd += dot(d, d)
    end
    msd /= M
    @test isapprox(msd, 6t; rtol=0.05)   # 6t = 2·d·D₀·t with d=3, D₀=1
end

@testset "passive chain CM diffusion D = 1/N" begin
    N = 10
    dt = 0.001
    nsteps = 500
    M = 3000
    t = nsteps * dt
    msd_cm = 0.0
    for m in 1:M
        rng = Xoshiro(UInt64(1000 + m))
        pos = initialize_chain(N; mode=:line, rng=rng)
        sys = System(pos; bonded=HarmonicBond(100.0, 1.0),
                     pair=SoftRepulsive(100.0, 1.0), activity=NoActivity())
        cm0 = center_of_mass(sys.positions)
        for _ in 1:nsteps
            bd_step!(sys, dt, rng)
        end
        d = center_of_mass(sys.positions) - cm0
        msd_cm += dot(d, d)
    end
    msd_cm /= M
    # CM MSD = 2·d·D_cm·t = 6·(1/N)·t exactly (internal forces cancel in CM).
    @test isapprox(msd_cm, 6t / N; rtol=0.06)
end

@testset "determinism and replica independence" begin
    N = 20
    dt = 1e-3
    make() = (rng = Xoshiro(0xABCDEF);
              pos = initialize_chain(N; mode=:line, rng=Xoshiro(7));
              (System(pos; bonded=HarmonicBond(100.0, 1.0),
                      pair=SoftRepulsive(100.0, 1.0),
                      activity=TangentialActivity(5.0)), rng))

    sysA, rngA = make(); sysB, rngB = make()
    for _ in 1:2000
        bd_step!(sysA, dt, rngA); bd_step!(sysB, dt, rngB)
    end
    @test sysA.positions == sysB.positions    # same seed ⇒ identical

    # Different replica streams ⇒ different trajectory.
    pos = initialize_chain(N; mode=:line, rng=Xoshiro(7))
    sysC = System(pos; bonded=HarmonicBond(100.0, 1.0),
                  pair=SoftRepulsive(100.0, 1.0), activity=TangentialActivity(5.0))
    rngC = replica_rng(UInt64(0xABCDEF), 2)
    for _ in 1:2000
        bd_step!(sysC, dt, rngC)
    end
    @test sysC.positions != sysA.positions
end

# Small helper: equilibrate then average R_G over production for one chain.
function mean_rg(N, dt, n_equil, n_prod, Pe, seed)
    rng = Xoshiro(UInt64(seed))
    pos = initialize_chain(N; mode=:line, rng=rng)
    sys = System(pos; bonded=HarmonicBond(100.0, 1.0),
                 pair=SoftRepulsive(100.0, 1.0), activity=TangentialActivity(Pe))
    for _ in 1:n_equil
        bd_step!(sys, dt, rng)
    end
    acc = 0.0; n = 0
    for k in 1:n_prod
        bd_step!(sys, dt, rng)
        if k % 20 == 0
            acc += radius_of_gyration(sys.positions); n += 1
        end
    end
    return acc / n
end

@testset "dt convergence of ⟨R_G⟩ (coarse)" begin
    N = 12; Pe = 5.0
    # Match physical time: halve dt, double steps.
    rg1 = mean_rg(N, 1e-3, 40_000, 120_000, Pe, 11)
    rg2 = mean_rg(N, 5e-4, 80_000, 240_000, Pe, 11)
    @test isapprox(rg1, rg2; rtol=0.12)      # statistical, so a loose bound
end
