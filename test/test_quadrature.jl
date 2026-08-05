# Quadrature weights used by analysis/thermo_integration.jl. These are the only
# genuinely new numerics in the TI path — everything else is a sum of energies —
# so they get exact-integral checks.
#
# The module is included at the top level of runtests.jl (a `module` cannot be
# defined inside the @testset block) and reached here as Main.ThermoIntegration.

const TI = Main.ThermoIntegration

@testset "TI quadrature weights" begin
    simp = TI.simpson_weights
    trap = TI.trapezoid_weights

    uniform_odd = collect(range(0.0, 2.0; length=7))    # even interval count
    uniform_even = collect(range(0.0, 2.0; length=8))   # trailing trapezoid panel
    nonuniform = [0.0, 0.1, 0.35, 0.6, 1.0, 1.5, 2.0]

    # Weights must reproduce ∫1 dx = interval length, whatever the grid.
    for x in (uniform_odd, uniform_even, nonuniform)
        @test sum(simp(x)) ≈ x[end] - x[1]
        @test sum(trap(x)) ≈ x[end] - x[1]
    end

    # On a uniform grid with an even interval count, composite Simpson is exact
    # through cubics. This is the case the generated (uniform-in-s) grid hits,
    # which is why ti.toml prefers an odd n_points.
    for (f, exact) in ((x -> x^2, 8 / 3), (x -> x^3, 4.0), (x -> 1 + 2x, 6.0))
        @test sum(simp(uniform_odd) .* f.(uniform_odd)) ≈ exact
    end

    # On a non-uniform grid the panel formula stays exact for quadratics; cubics
    # pick up a small error, so an explicit k_values list integrates slightly
    # less accurately than the generated grid.
    @test sum(simp(nonuniform) .* (nonuniform .^ 2)) ≈ 8 / 3
    @test isapprox(sum(simp(nonuniform) .* (nonuniform .^ 3)), 4.0; atol=1e-2)

    # Simpson must beat trapezoid on a convex integrand — the property the
    # reported |Simpson - trapezoid| quadrature-error estimate leans on.
    @test abs(sum(simp(uniform_odd) .* (uniform_odd .^ 2)) - 8 / 3) <
          abs(sum(trap(uniform_odd) .* (uniform_odd .^ 2)) - 8 / 3)

    # Trapezoid on a linear integrand is exact.
    @test sum(trap(nonuniform) .* (1 .+ 2 .* nonuniform)) ≈ 6.0

    # Degenerate and malformed grids.
    @test simp(Float64[]) == Float64[]
    @test simp([1.0]) == [0.0]
    @test simp([0.0, 1.0]) ≈ [0.5, 0.5]         # 2 points ⇒ trapezoid
    @test_throws ErrorException simp([0.0, 1.0, 1.0, 2.0])   # not strictly increasing
end
