# Test 4 (§10): gyration-tensor identities, plus checks on the other
# observables and a passive-baseline bond-length sanity check (§10 check 7).

using Statistics

@testset "gyration tensor identities" begin
    rng = Xoshiro(42)
    pos = [randn(rng, ActivePolyBD.Vec3) for _ in 1:15]

    S = gyration_tensor(pos)
    @test isapprox(S, S')                          # symmetric
    @test isapprox(tr(S), radius_of_gyration_sq(pos); rtol=1e-12)
    @test isapprox(radius_of_gyration(pos)^2, radius_of_gyration_sq(pos); rtol=1e-12)

    λ1, λ2, λ3 = gyration_eigenvalues(pos)
    @test λ1 >= λ2 >= λ3 >= -1e-10                 # non-negative, ordered
    @test isapprox(λ1 + λ2 + λ3, radius_of_gyration_sq(pos); rtol=1e-10)

    # Known configuration: two beads at ±(a,0,0). cm=0, R_G² = a².
    a = 2.0
    two = [ActivePolyBD.Vec3(a, 0, 0), ActivePolyBD.Vec3(-a, 0, 0)]
    @test isapprox(radius_of_gyration_sq(two), a^2; rtol=1e-12)
    @test isapprox(center_of_mass(two), zero(ActivePolyBD.Vec3); atol=1e-12)
end

@testset "asphericity: sphere vs rod" begin
    # Collinear points along x → rod → A = 1.
    rod = [ActivePolyBD.Vec3(x, 0, 0) for x in -3:3]
    acc = AsphericityAccumulator()
    ActivePolyBD.accumulate!(acc, rod)
    @test isapprox(asphericity(acc), 1.0; atol=1e-8)

    # Cubically symmetric cloud (±1 on each axis) → isotropic → A = 0.
    iso = [ActivePolyBD.Vec3(1,0,0), ActivePolyBD.Vec3(-1,0,0),
           ActivePolyBD.Vec3(0,1,0), ActivePolyBD.Vec3(0,-1,0),
           ActivePolyBD.Vec3(0,0,1), ActivePolyBD.Vec3(0,0,-1)]
    acc2 = AsphericityAccumulator()
    ActivePolyBD.accumulate!(acc2, iso)
    @test isapprox(asphericity(acc2), 0.0; atol=1e-8)
end

@testset "end-to-end and backbone angles" begin
    pos = [ActivePolyBD.Vec3(Float64(i), 0, 0) for i in 1:6]
    @test end_to_end(pos) == ActivePolyBD.Vec3(5, 0, 0)
    cosθ = backbone_cosangles(pos)
    @test length(cosθ) == 4
    @test all(c -> isapprox(c, 1.0), cosθ)         # straight chain ⇒ cosθ = 1
end

@testset "passive baseline: bond lengths near b (equipartition)" begin
    N = 20; dt = 1e-3
    rng = Xoshiro(99)
    pos = initialize_chain(N; mode=:line, rng=rng)
    sys = System(pos; bonded=HarmonicBond(100.0, 1.0),
                 pair=SoftRepulsive(100.0, 1.0), activity=NoActivity())
    for _ in 1:100_000
        bd_step!(sys, dt, rng)
    end
    # ⟨(r_ij − b)²⟩ ≈ k_BT/K_sp = 1/100 per bond (1D harmonic equipartition).
    var_acc = 0.0; n = 0
    for _ in 1:20_000
        bd_step!(sys, dt, rng)
        for i in 1:N-1
            var_acc += (norm(sys.positions[i] - sys.positions[i+1]) - 1.0)^2
            n += 1
        end
    end
    mean_var = var_acc / n
    @test isapprox(mean_var, 1 / 100; rtol=0.25)   # loose: soft-core perturbs slightly
    @test radius_of_gyration(sys.positions) > 0
end
