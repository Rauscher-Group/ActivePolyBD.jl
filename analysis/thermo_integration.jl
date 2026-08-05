# Thermodynamic integration: free-energy difference between the bonded-only
# (phantom) chain and the full passive model with soft-core excluded volume.
#
# The soft repulsion V(r) = ½k(r-r_c)² is finite at r = 0 and linear in k, so k
# itself is a valid coupling parameter — no soft-core singularity at the k → 0
# endpoint. Then
#
#     ΔF = F(k_max) - F(0) = ∫₀^{k_max} ⟨∂U/∂k⟩_k dk,
#
# and the integrand is the `dUdc` column the simulator logs.
#
# Usage (standalone):  julia analysis/thermo_integration.jl <ti_out_dir> [a]

module ThermoIntegration

using Printf
using Statistics

include(joinpath(@__DIR__, "rg_distribution.jl"))
using .RgDistribution: read_column

# scalar log: step time Rg cm_x cm_y cm_z Re2 U_bond U_nb dUdc
const COL_RG = 3
const COL_UBOND = 8
const COL_UNB = 9
const COL_DUDC = 10

"""
    read_column_per_file(dir, col) -> Vector{Vector{Float64}}

Column `col` from each `obs_r*.dat` log in `dir`, kept **separate per replica**.
`RgDistribution.read_column` pools them, which is right for a histogram but
wrong for an error bar: samples within a replica are autocorrelated, whereas
replicas are fully independent (each has its own RNG stream), so the
replica-to-replica scatter is the honest uncertainty on the mean.
"""
function read_column_per_file(dir::AbstractString, col::Integer)
    out = Vector{Float64}[]
    for f in sort(readdir(dir))
        (startswith(f, "obs") && endswith(f, ".dat")) || continue
        vals = Float64[]
        for line in eachline(joinpath(dir, f))
            (isempty(line) || startswith(line, "#")) && continue
            cols = split(line)
            length(cols) >= col || continue
            push!(vals, parse(Float64, cols[col]))
        end
        isempty(vals) || push!(out, vals)
    end
    return out
end

"""
    k_label(dir) -> Float64

Read the coupling value written by the TI driver (`k.txt`), falling back to
decoding the directory name (`k_3p657` → 3.657).
"""
function k_label(dir::AbstractString)
    kf = joinpath(dir, "k.txt")
    isfile(kf) && return parse(Float64, strip(read(kf, String)))
    name = basename(dir)
    startswith(name, "k_") || return NaN
    return parse(Float64, replace(name[3:end], "p" => ".", "m" => "-"))
end

# --- Quadrature ----------------------------------------------------------

"""
    simpson_weights(x) -> Vector{Float64}

Composite-Simpson quadrature weights on a possibly **non-uniform** grid `x`
(Cartwright's form), so `∫ f dx ≈ dot(w, f)`. Panels are taken in consecutive
triples; a trailing odd interval falls back to the trapezoid rule.

Returning weights rather than a scalar lets the caller propagate per-point
statistical errors through the same quadrature.
"""
function simpson_weights(x::AbstractVector{<:Real})
    n = length(x)
    w = zeros(Float64, n)
    n >= 2 || return w
    i = 1
    while i + 2 <= n
        h0 = x[i+1] - x[i]
        h1 = x[i+2] - x[i+1]
        (h0 > 0 && h1 > 0) || error("TI grid must be strictly increasing; got $(x[i:i+2])")
        s = h0 + h1
        w[i]   += s / 6 * (2 - h1 / h0)
        w[i+1] += s / 6 * (s^2 / (h0 * h1))
        w[i+2] += s / 6 * (2 - h0 / h1)
        i += 2
    end
    if i + 1 == n                      # odd interval count: trapezoid the tail
        h = x[n] - x[n-1]
        w[n-1] += h / 2
        w[n]   += h / 2
    end
    return w
end

"""
    trapezoid_weights(x) -> Vector{Float64}
"""
function trapezoid_weights(x::AbstractVector{<:Real})
    n = length(x)
    w = zeros(Float64, n)
    for i in 1:n-1
        h = x[i+1] - x[i]
        w[i]   += h / 2
        w[i+1] += h / 2
    end
    return w
end

# --- Analysis ------------------------------------------------------------

"""
    analyze(out_dir; a=1.0) -> NamedTuple

Read every `k_*` subdirectory under `out_dir` and integrate ⟨∂U/∂k⟩ over k.

The integration is done in the substituted variable `s = ln(1 + k/a)`, where
`dk = (k + a) ds`:

    ΔF = ∫ ⟨∂U/∂k⟩ dk = ∫ (k + a)·⟨∂U/∂k⟩ ds

⟨∂U/∂k⟩ is finite at k = 0 but decays roughly as C/k at large k (the energy
penalty saturates near k_BT per contact), so in raw k it is flat at one end and
power-law at the other and no modest uniform grid resolves both. The
substituted integrand `(k+a)⟨∂U/∂k⟩` is bounded at *both* ends — `a⟨∂U/∂k⟩(0)`
at k = 0 and → C as k → ∞ — which is what makes a dozen-odd points enough.

`a` is a **quadrature knob, not a physical parameter**: it is a change of
variables and cannot change ΔF, only how well a given grid resolves the
integrand. Set it near the crossover scale (a ≈ 1 in reduced units).

Writes `ti_integrand.csv` and `ti_result.txt` under `out_dir`, prints a summary
table and an ASCII curve, and saves a PNG if `Plots` can be loaded.
"""
function analyze(out_dir::AbstractString; a::Real=1.0)
    a > 0 || throw(ArgumentError("substitution scale a must be > 0; got $a"))
    subdirs = sort(filter(d -> isdir(joinpath(out_dir, d)) && startswith(d, "k_"),
                          readdir(out_dir)))
    isempty(subdirs) && error("no k_* subdirectories under $out_dir")

    ks = Float64[]
    means = Float64[]
    stderrs = Float64[]
    unbs = Float64[]
    ubonds = Float64[]
    rgs = Float64[]
    nsamples = Int[]
    nreplicas = Int[]

    for d in subdirs
        full = joinpath(out_dir, d)
        per_replica = read_column_per_file(full, COL_DUDC)
        if isempty(per_replica)
            @warn "no dUdc samples in $full; logs predate the energy columns" dir = full
            continue
        end
        pooled = reduce(vcat, per_replica)
        rmeans = mean.(per_replica)
        push!(ks, k_label(full))
        push!(means, mean(pooled))
        # Independent replicas ⇒ standard error of their means. With a single
        # replica there is no scatter to measure; report NaN rather than 0,
        # which would understate the uncertainty as "exact".
        push!(stderrs, length(rmeans) < 2 ? NaN : std(rmeans) / sqrt(length(rmeans)))
        push!(unbs, mean(read_column(full, COL_UNB)))
        push!(ubonds, mean(read_column(full, COL_UBOND)))
        push!(rgs, mean(read_column(full, COL_RG)))
        push!(nsamples, length(pooled))
        push!(nreplicas, length(per_replica))
    end
    length(ks) >= 2 || error("need at least 2 k points to integrate; got $(length(ks))")

    ord = sortperm(ks)
    ks, means, stderrs = ks[ord], means[ord], stderrs[ord]
    unbs, ubonds, rgs = unbs[ord], ubonds[ord], rgs[ord]
    nsamples, nreplicas = nsamples[ord], nreplicas[ord]

    s = @. log(1 + ks / a)
    integrand = @. (ks + a) * means            # g(s), bounded at both ends

    w_simp = simpson_weights(s)
    w_trap = trapezoid_weights(s)
    dF = sum(w_simp .* integrand)
    dF_trap = sum(w_trap .* integrand)
    quad_err = abs(dF - dF_trap)               # Simpson vs trapezoid spread

    # Statistical error through the same weights. NaN entries (single-replica
    # points) are skipped so one such point does not poison the whole bar.
    stat_var = 0.0
    n_nan = 0
    for i in eachindex(ks)
        e = w_simp[i] * (ks[i] + a) * stderrs[i]
        isnan(e) ? (n_nan += 1) : (stat_var += e^2)
    end
    stat_err = sqrt(stat_var)

    # --- CSV -------------------------------------------------------------
    csv_path = joinpath(out_dir, "ti_integrand.csv")
    open(csv_path, "w") do io
        println(io, "k,s,mean_dUdc,stderr_dUdc,integrand,mean_Unb,mean_Ubond,mean_Rg,nsamples,nreplicas")
        for i in eachindex(ks)
            @printf(io, "%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d,%d\n",
                    ks[i], s[i], means[i], stderrs[i], integrand[i],
                    unbs[i], ubonds[i], rgs[i], nsamples[i], nreplicas[i])
        end
    end

    res_path = joinpath(out_dir, "ti_result.txt")
    open(res_path, "w") do io
        println(io, "Thermodynamic integration: phantom chain (k=0) -> full excluded volume")
        println(io, "  DeltaF          = ", @sprintf("%.6g", dF), " k_BT")
        println(io, "  statistical err = ", @sprintf("%.3g", stat_err),
                n_nan > 0 ? "  ($n_nan point(s) had <2 replicas and were omitted)" : "")
        println(io, "  quadrature err  = ", @sprintf("%.3g", quad_err),
                "  (|Simpson - trapezoid|)")
        println(io, "  trapezoid       = ", @sprintf("%.6g", dF_trap))
        println(io, "  substitution    = s = ln(1 + k/a), a = ", @sprintf("%g", a))
        println(io, "  k grid          = ", join((@sprintf("%g", k) for k in ks), " "))
    end

    _report(ks, s, means, stderrs, integrand, unbs, ubonds, rgs, nreplicas,
            dF, stat_err, quad_err, dF_trap, a, csv_path, res_path)
    _try_plot(out_dir, ks, s, means, integrand)

    return (; ks, s, means, stderrs, integrand, dF, stat_err, quad_err)
end

function _report(ks, s, means, stderrs, integrand, unbs, ubonds, rgs, nreplicas,
                 dF, stat_err, quad_err, dF_trap, a, csv_path, res_path)
    println("\nTI integrand vs coupling k  (a = ", @sprintf("%g", a), "):")
    println(rpad("k", 10), rpad("s", 9), rpad("⟨∂U/∂k⟩", 12), rpad("stderr", 11),
            rpad("(k+a)⟨∂U/∂k⟩", 15), rpad("⟨U_nb⟩", 11), rpad("⟨U_bond⟩", 11),
            rpad("⟨R_G⟩", 10), "nrep")
    for i in eachindex(ks)
        println(rpad(@sprintf("%g", ks[i]), 10),
                rpad(@sprintf("%.3f", s[i]), 9),
                rpad(@sprintf("%.5f", means[i]), 12),
                rpad(@sprintf("%.2e", stderrs[i]), 11),
                rpad(@sprintf("%.5f", integrand[i]), 15),
                rpad(@sprintf("%.4f", unbs[i]), 11),
                rpad(@sprintf("%.4f", ubonds[i]), 11),
                rpad(@sprintf("%.4f", rgs[i]), 10),
                nreplicas[i])
    end

    # ⟨U_bond⟩ is a k-independent equilibrium property, so drift across k means
    # the sampling or the energy loop is wrong and is worth flagging loudly.
    #
    # The test is *flatness*, not the absolute value. The ideal is ½ k_BT per
    # bond — (N-1)/2, only the radial mode being confined — but the measured
    # value sits above it by two known, k-independent corrections: the r²dr
    # Jacobian (O(1/(k_bond r₀²)), a couple of percent) and the O(dt) bias of
    # Euler–Maruyama, which inflates the spring variance by ≈1/(1 - k_rel·dt/2)
    # with k_rel = 2k_bond — ~11% at k_bond = 100, dt = 1e-3. Both cancel out of
    # ΔF, which depends only on the non-bonded coupling.
    spread = maximum(ubonds) - minimum(ubonds)
    scale = abs(mean(ubonds))
    if scale > 0 && spread / scale > 0.05
        @warn "⟨U_bond⟩ varies by $(round(100*spread/scale; digits=1))% across k; \
               it should be independent of the non-bonded coupling" ubonds
    end

    println("\nSubstituted integrand (k+a)⟨∂U/∂k⟩ vs s — flat-ish means well resolved:")
    m = maximum(integrand; init=0.0)
    for i in eachindex(s)
        bar = (isfinite(m) && m > 0) ?
              repeat("█", round(Int, 50 * clamp(integrand[i] / m, 0, 1))) : ""
        println(rpad(@sprintf("%8.3f", s[i]), 10), bar)
    end

    println("\nΔF = F(k=", @sprintf("%g", ks[end]), ") - F(k=", @sprintf("%g", ks[1]), ") = ",
            @sprintf("%.6g", dF), " ± ", @sprintf("%.3g", stat_err), " k_BT (statistical)")
    println("    quadrature error ≈ ", @sprintf("%.3g", quad_err),
            "  (trapezoid gives ", @sprintf("%.6g", dF_trap), ")")
    if isfinite(quad_err) && isfinite(stat_err) && quad_err > stat_err
        println("    NOTE: quadrature error exceeds the statistical error — add k points ",
                "where the\n          integrand curve above changes fastest, then re-run analyze.")
    end
    println("\nWrote:\n  ", csv_path, "\n  ", res_path)
    return nothing
end

# Optional PNG if a plotting backend is installed. Never a hard dep.
function _try_plot(out_dir, ks, s, means, integrand)
    try
        @eval import Plots
        plt = Base.invokelatest(Plots.plot; layout=(1, 2), size=(900, 380), legend=false)
        Base.invokelatest(Plots.plot!, plt, ks, means;
                          subplot=1, xlabel="k", ylabel="⟨∂U/∂k⟩", marker=:circle, lw=2)
        Base.invokelatest(Plots.plot!, plt, s, integrand;
                          subplot=2, xlabel="s = ln(1+k/a)", ylabel="(k+a)⟨∂U/∂k⟩",
                          marker=:circle, lw=2)
        png = joinpath(out_dir, "ti_integrand.png")
        Base.invokelatest(Plots.savefig, plt, png)
        println("  ", png)
    catch err
        @info "Plots backend unavailable; wrote CSV/ASCII only (install Plots for a PNG)." exception = (err,)
    end
    return nothing
end

end # module ThermoIntegration

# Allow standalone invocation.
if abspath(PROGRAM_FILE) == @__FILE__
    if !(1 <= length(ARGS) <= 2)
        println(stderr, "usage: julia analysis/thermo_integration.jl <ti_out_dir> [a]")
        exit(1)
    end
    ThermoIntegration.analyze(ARGS[1]; a=length(ARGS) == 2 ? parse(Float64, ARGS[2]) : 1.0)
end
