using ActivePolyBD
using Test
using Random
using LinearAlgebra
using StaticArrays

@testset "ActivePolyBD.jl" begin
    include("test_potentials.jl")       # Test 1: force ↔ energy FD
    include("test_forces_activity.jl")  # force balance, exclusions, activity
    include("test_integrator.jl")       # Tests 2,3,5,6: MSD, CM D=1/N, determinism, dt
    include("test_observables.jl")      # Test 4: gyration identities, baseline
    include("test_config_io.jl")        # config, XYZ/log, checkpoint restart
    include("test_headline.jl")         # Check 8: ⟨R_G⟩ decreases with Pe
end
