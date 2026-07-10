# Qualitative check 8 (§10): the headline trend. At fixed N, ⟨R_G⟩ should
# decrease as Pe increases (coil → globule-like), per Fig. 2a–b. Kept short
# and with a generous margin so it is a trend check, not a fitted comparison.

using Statistics

function equilibrated_mean_rg(N, Pe; dt=1e-3, n_equil=60_000, n_prod=200_000,
                              stride=25, seed=2024)
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
        if k % stride == 0
            acc += radius_of_gyration(sys.positions); n += 1
        end
    end
    return acc / n
end

@testset "headline trend: ⟨R_G⟩ shrinks with Pe" begin
    N = 20
    rg0 = equilibrated_mean_rg(N, 0.0)
    rg_hi = equilibrated_mean_rg(N, 100.0)
    @info "headline trend" N rg_passive=rg0 rg_active=rg_hi
    @test rg_hi < rg0                    # active chain is more compact
    @test rg_hi < 0.85 * rg0             # and clearly so at large Pe
end
