# Force-loop and activity correctness: internal-force balance, exclusions,
# passive ends, and the skip-one tangent (§2.3, §2.6, §14).

@testset "conservative internal forces sum to zero" begin
    rng = Xoshiro(3)
    pos = initialize_chain(8; mode=:saw, rng=rng)
    sys = System(pos; bonded=HarmonicBond(100.0, 1.0),
                 pair=SoftRepulsive(100.0, 1.0), activity=NoActivity())
    compute_forces!(sys)
    # Bonded + non-bonded are internal (Newton's third law) ⇒ Σ F = 0.
    @test isapprox(sum(sys.forces), zero(ActivePolyBD.Vec3); atol=1e-10)
end

@testset "active force: passive ends, skip-one tangent, Pe magnitude" begin
    # Zig-zag so tangents are well defined and non-trivial.
    pos = [ActivePolyBD.Vec3(0,0,0), ActivePolyBD.Vec3(1,0.3,0),
           ActivePolyBD.Vec3(2,0,0), ActivePolyBD.Vec3(3,0.3,0),
           ActivePolyBD.Vec3(4,0,0)]
    f = zeros(ActivePolyBD.Vec3, length(pos))
    ActivePolyBD.add_active_forces!(f, pos, TangentialActivity(7.0))

    @test f[1] == zero(ActivePolyBD.Vec3)          # end passive
    @test f[end] == zero(ActivePolyBD.Vec3)        # end passive
    for i in 2:length(pos)-1
        @test isapprox(norm(f[i]), 7.0; rtol=1e-12)  # |f_act| = Pe
        t = pos[i+1] - pos[i-1]                       # skip-one, not adjacent bond
        @test isapprox(f[i], 7.0 * t / norm(t); rtol=1e-12)
    end
end

@testset "pair loop excludes bonded neighbors and respects cutoff" begin
    # Three beads in a line at spacing 0.5 (< b): 1–2 and 2–3 are bonded
    # (excluded); 1–3 are separated by 1.0 = cutoff ⇒ no force. Net pair
    # force must be exactly zero.
    pos = [ActivePolyBD.Vec3(0,0,0), ActivePolyBD.Vec3(0.5,0,0), ActivePolyBD.Vec3(1.0,0,0)]
    f = zeros(ActivePolyBD.Vec3, 3)
    ActivePolyBD.add_pair_forces!(f, pos, SoftRepulsive(100.0, 1.0), AllPairs())
    @test all(fi -> isapprox(fi, zero(ActivePolyBD.Vec3); atol=1e-12), f)

    # Now 1–3 overlap (distance 0.9 < 1.0) ⇒ they repel each other, 2 untouched.
    pos2 = [ActivePolyBD.Vec3(0,0,0), ActivePolyBD.Vec3(0.45,0,0), ActivePolyBD.Vec3(0.9,0,0)]
    f2 = zeros(ActivePolyBD.Vec3, 3)
    ActivePolyBD.add_pair_forces!(f2, pos2, SoftRepulsive(100.0, 1.0), AllPairs())
    @test f2[1][1] < 0                               # bead 1 pushed −x
    @test f2[3][1] > 0                               # bead 3 pushed +x
    @test isapprox(f2[2], zero(ActivePolyBD.Vec3); atol=1e-12)
    @test isapprox(f2[1] + f2[3], zero(ActivePolyBD.Vec3); atol=1e-12)
end
