# Observables (reduced units, §6). Cheap for N ≤ 40, computed on demand.
# R_G is the primary observable and is logged during production.

"""
    center_of_mass(pos) -> Vec3
"""
@inline center_of_mass(pos::Vector{Vec3}) = sum(pos) / length(pos)

"""
    gyration_tensor(pos) -> SMatrix{3,3}

`S = (1/N) Σ_i (r_i - cm)(r_i - cm)ᵀ`.
"""
function gyration_tensor(pos::Vector{Vec3})
    cm = center_of_mass(pos)
    S = zero(SMatrix{3,3,Float64})
    @inbounds for r in pos
        d = r - cm
        S += d * d'
    end
    return S / length(pos)
end

"""
    radius_of_gyration(pos) -> Float64

`R_G = sqrt(tr S)`. Primary observable.
"""
@inline function radius_of_gyration(pos::Vector{Vec3})
    cm = center_of_mass(pos)
    s2 = 0.0
    @inbounds for r in pos
        d = r - cm
        s2 += dot(d, d)
    end
    return sqrt(s2 / length(pos))
end

"""
    radius_of_gyration_sq(pos) -> Float64

`R_G² = tr S`, without the square root.
"""
@inline function radius_of_gyration_sq(pos::Vector{Vec3})
    cm = center_of_mass(pos)
    s2 = 0.0
    @inbounds for r in pos
        d = r - cm
        s2 += dot(d, d)
    end
    return s2 / length(pos)
end

"""
    gyration_eigenvalues(pos) -> (λ1, λ2, λ3)  with λ1 ≥ λ2 ≥ λ3 ≥ 0
"""
function gyration_eigenvalues(pos::Vector{Vec3})
    S = gyration_tensor(pos)
    λ = eigvals(Symmetric(Matrix(S)))
    return (λ[3], λ[2], λ[1])
end

"""
    end_to_end(pos) -> Vec3
"""
@inline end_to_end(pos::Vector{Vec3}) = pos[end] - pos[1]

"""
    backbone_cosangles(pos) -> Vector{Float64}

`cosθ_i = û_{i-1,i} · û_{i,i+1}` at interior monomers i = 2 … N-1.
"""
function backbone_cosangles(pos::Vector{Vec3})
    N = length(pos)
    out = Vector{Float64}(undef, max(N - 2, 0))
    @inbounds for i in 2:N-1
        u1 = pos[i] - pos[i-1]
        u2 = pos[i+1] - pos[i]
        out[i-1] = dot(u1, u2) / (norm(u1) * norm(u2))
    end
    return out
end

"""
    AsphericityAccumulator()

Accumulates ⟨Tr²⟩ and ⟨3M⟩ separately over frames so the asphericity ratio
`A = ⟨Tr² - 3M⟩ / ⟨Tr²⟩` is formed from ensemble averages, never per-frame
ratios (§14). `Tr = λ1+λ2+λ3`, `M = λ1λ2 + λ1λ3 + λ2λ3`.
"""
mutable struct AsphericityAccumulator
    sum_tr2::Float64
    sum_3m::Float64
    n::Int
end
AsphericityAccumulator() = AsphericityAccumulator(0.0, 0.0, 0)

function accumulate!(acc::AsphericityAccumulator, pos::Vector{Vec3})
    λ1, λ2, λ3 = gyration_eigenvalues(pos)
    tr = λ1 + λ2 + λ3
    M = λ1 * λ2 + λ1 * λ3 + λ2 * λ3
    acc.sum_tr2 += tr^2
    acc.sum_3m += 3 * M
    acc.n += 1
    return acc
end

"""
    asphericity(acc) -> Float64

`A = ⟨Tr² - 3M⟩ / ⟨Tr²⟩ = (⟨Tr²⟩ - ⟨3M⟩)/⟨Tr²⟩`. A = 0 for a sphere,
A = 1 for a rod.
"""
function asphericity(acc::AsphericityAccumulator)
    acc.n == 0 && return NaN
    return (acc.sum_tr2 - acc.sum_3m) / acc.sum_tr2
end
