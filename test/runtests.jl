using ActivePolyBD
using Test
using Random
using LinearAlgebra
using StaticArrays

# Top level: `module ThermoIntegration` cannot be defined inside the @testset.
include(joinpath(@__DIR__, "..", "analysis", "thermo_integration.jl"))

@testset "ActivePolyBD.jl" begin
    include("test_potentials.jl")       # Test 1: force ↔ energy FD
    include("test_forces_activity.jl")  # force balance, exclusions, activity
    include("test_integrator.jl")       # Tests 2,3,5,6: MSD, CM D=1/N, determinism, dt
    include("test_observables.jl")      # Test 4: gyration identities, baseline
    include("test_energy.jl")           # energy sums ↔ force loops, TI coupling
    include("test_quadrature.jl")       # TI Simpson/trapezoid weights
    include("test_config_io.jl")        # config, XYZ/log, checkpoint restart
    include("test_headline.jl")         # Check 8: ⟨R_G⟩ decreases with Pe
end
