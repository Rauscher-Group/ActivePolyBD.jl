# Test 1 (§10): analytic -V'(r) matches central finite difference of energy(r)
# for every potential. Catches sign and factor errors.

@testset "force ↔ energy consistency (finite difference)" begin
    fd(f, r; h=1e-6) = (f(r + h) - f(r - h)) / (2h)

    potentials = Any[
        HarmonicBond(100.0, 1.0),
        FENEBond(30.0, 1.5),
        SoftRepulsive(100.0, 1.0),
        WCA(1.0, 1.0),
    ]

    for p in potentials
        # Sample r inside each potential's valid/active range.
        rs = if p isa FENEBond
            range(0.1, 0.95 * p.R0; length=25)
        elseif p isa SoftRepulsive
            range(0.3, 0.99 * p.cutoff; length=25)   # active branch only
        elseif p isa WCA
            range(0.85, 0.99 * ActivePolyBD.cutoff(p); length=25)
        else
            range(0.4, 1.8; length=25)
        end
        for r in rs
            analytic = dVdr(p, r)
            numeric = fd(x -> energy(p, x), r)
            @test isapprox(analytic, numeric; rtol=1e-6, atol=1e-6)
        end
    end

    # SoftRepulsive: zero force and energy at/above cutoff (C¹ turn-on).
    sr = SoftRepulsive(100.0, 1.0)
    @test dVdr(sr, 1.0) == 0.0
    @test energy(sr, 1.0) == 0.0
    @test dVdr(sr, 1.5) == 0.0
    @test dVdr(sr, 0.5) < 0.0        # repulsive: pushes apart
end
