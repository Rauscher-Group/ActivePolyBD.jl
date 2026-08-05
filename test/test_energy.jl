# System-level potential energies: exclusion/cutoff bookkeeping, linearity in
# the coupling parameter, and — the one that matters — that the energy sums in
# observables.jl differentiate to the force loops in forces.jl. The two loops
# are written out separately, so nothing but this test keeps them in step.

const V3 = ActivePolyBD.Vec3

@testset "potential energies" begin

    @testset "exclusion and cutoff" begin
        sr = SoftRepulsive(100.0, 1.0)
        nb = AllPairs()

        # A bonded pair inside the cutoff contributes nothing: |i-j| == 1 is
        # excluded, its interaction being the bond's job.
        two = [V3(0, 0, 0), V3(0.5, 0, 0)]
        @test pair_energy(two, sr, nb) == 0.0

        # 1-3 is not excluded. Place beads 1 and 3 at separation r with bead 2
        # off the axis, so exactly one pair contributes.
        r = 0.6
        three = [V3(0, 0, 0), V3(r / 2, 5.0, 0), V3(r, 0, 0)]
        @test pair_energy(three, sr, nb) ≈ energy(sr, r)

        # At and beyond the cutoff: exactly zero (the test is strict `r < rc`).
        @test pair_energy([V3(0, 0, 0), V3(0.5, 5.0, 0), V3(1.0, 0, 0)], sr, nb) == 0.0
        @test pair_energy([V3(0, 0, 0), V3(0.5, 5.0, 0), V3(1.5, 0, 0)], sr, nb) == 0.0

        # Bonded energy counts all N-1 bonds and nothing else.
        hb = HarmonicBond(100.0, 1.0)
        @test bonded_energy(three, hb) ≈
              energy(hb, norm(three[2] - three[1])) + energy(hb, norm(three[3] - three[2]))

        # VerletList must give identical energies to AllPairs, as it does for
        # forces, or swapping strategies would silently change results.
        rng = Xoshiro(7)
        pos = [randn(rng, V3) for _ in 1:10]
        @test pair_energy(pos, sr, VerletList(0.3)) == pair_energy(pos, sr, nb)
    end

    @testset "coupling linearity" begin
        nb = AllPairs()
        rng = Xoshiro(11)
        # Scale a random walk down so beads genuinely overlap and the sum is
        # non-trivial.
        pos = V3[V3(0, 0, 0)]
        for _ in 2:12
            push!(pos, pos[end] + 0.7 * normalize(randn(rng, V3)))
        end

        for p in (SoftRepulsive(100.0, 1.0), WCA(1.0, 1.0))
            @test coupling(with_coupling(p, 3.5)) == 3.5
            unit = pair_energy(pos, with_coupling(p, 1.0), nb)
            @test unit > 0                       # configuration must exercise the sum
            for c in (0.0, 0.5, 1.0, 7.0, 100.0)
                @test pair_energy(pos, with_coupling(p, c), nb) ≈ c * unit
            end
            # Coupling must not move the cutoff, or which pairs interact would
            # change with c and ∂U/∂c would pick up a spurious term.
            @test ActivePolyBD.cutoff(with_coupling(p, 42.0)) == ActivePolyBD.cutoff(p)
        end
    end

    @testset "nonbonded_energies at c = 0 (phantom endpoint)" begin
        pos = [V3(0, 0, 0), V3(0.5, 5.0, 0), V3(0.6, 0, 0)]
        sys = System(pos; bonded=HarmonicBond(100.0, 1.0),
                     pair=SoftRepulsive(0.0, 1.0), activity=NoActivity())
        U_nb, dUdc = nonbonded_energies(sys)
        @test U_nb == 0.0            # no interaction at zero coupling ...
        @test dUdc > 0.0             # ... but the TI integrand is still finite

        # And at finite coupling the two agree with the direct sum.
        sys100 = System(copy(pos); bonded=HarmonicBond(100.0, 1.0),
                        pair=SoftRepulsive(100.0, 1.0), activity=NoActivity())
        U100, d100 = nonbonded_energies(sys100)
        @test U100 ≈ nonbonded_energy(sys100)
        @test U100 ≈ 100.0 * d100
        @test d100 ≈ dUdc            # the derivative is configuration-only
    end

    @testset "force ↔ energy consistency (system-wide FD)" begin
        # Numerically differentiate the *total* energy with respect to each
        # Cartesian coordinate and compare against the force loop. This is what
        # catches a divergence in exclusion or cutoff logic between the two.
        rng = Xoshiro(2024)
        N = 8
        pos = V3[V3(0, 0, 0)]
        for _ in 2:N
            push!(pos, pos[end] + 0.85 * normalize(randn(rng, V3)))
        end

        sr = SoftRepulsive(100.0, 1.0)
        hb = HarmonicBond(100.0, 1.0)
        nbs = AllPairs()
        h = 1e-6

        # Verify the configuration actually has non-bonded overlaps, or the
        # pair half of this test would pass vacuously.
        @test pair_energy(pos, sr, nbs) > 0

        for (U, addforces!) in (
            (p -> pair_energy(p, sr, nbs), (f, p) -> ActivePolyBD.add_pair_forces!(f, p, sr, nbs)),
            (p -> bonded_energy(p, hb), (f, p) -> ActivePolyBD.add_bonded_forces!(f, p, hb)),
        )
            forces = zeros(V3, N)
            addforces!(forces, pos)
            for i in 1:N, α in 1:3
                e = V3(ntuple(β -> β == α ? 1.0 : 0.0, 3))
                shifted = copy(pos)
                shifted[i] = pos[i] + h * e
                Uplus = U(shifted)
                shifted[i] = pos[i] - h * e
                Uminus = U(shifted)
                # F = -∇U
                @test isapprox(-(Uplus - Uminus) / (2h), forces[i][α];
                               rtol=1e-5, atol=1e-5)
            end
        end
    end

    @testset "equipartition baseline (short BD run)" begin
        # For V = ½k(r-r₀)² only the *radial* degree of freedom is confined —
        # the two angular ones are free — so the expectation is ½k_BT per bond,
        # i.e. ⟨U_bond⟩ = (N-1)/2, NOT 3(N-1)/2. Two known biases sit on top of
        # it, hence the loose tolerance (the same physics, and the same 0.25, as
        # the ⟨(r-b)²⟩ ≈ k_BT/K_sp check in test_observables):
        #   - the r²dr Jacobian, O(1/(k r₀²)), a couple of percent;
        #   - the O(dt) Euler–Maruyama bias, which inflates the spring variance
        #     by ≈1/(1 - k_rel·dt/2) with k_rel = 2k. dt = 1e-4 is used here to
        #     keep that ~1%; at the production dt = 1e-3 it is ~11%.
        rng = Xoshiro(31337)
        N = 20
        pos = initialize_chain(N; mode=:line, rng=rng)
        sys = System(pos; bonded=HarmonicBond(100.0, 1.0),
                     pair=SoftRepulsive(100.0, 1.0), activity=NoActivity())
        dt = 1e-4
        for _ in 1:100_000
            bd_step!(sys, dt, rng)
        end
        acc = 0.0
        nsamp = 0
        for step in 1:200_000
            bd_step!(sys, dt, rng)
            if step % 10 == 0
                acc += bonded_energy(sys)
                nsamp += 1
            end
        end
        @test isapprox(acc / nsamp, (N - 1) / 2; rtol=0.25)
    end
end
