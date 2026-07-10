# Headline analysis (§10a): read R_G logs across Pe, build P(R_G), and emit
# the overlay data + a ⟨R_G⟩-vs-Pe summary. Designed to run without a heavy
# plotting dependency: it always writes CSVs and an ASCII overlay to stdout,
# and additionally saves a PNG if a plotting backend (Plots) is available.
#
# Usage (standalone):  julia analysis/rg_distribution.jl <sweep_out_dir>

module RgDistribution

using Printf
using Statistics

"""
    read_rg(dir; warmup_frac=0.1) -> Vector{Float64}

Pool the R_G column (column 3) from every `obs_r*.dat` log in `dir`,
discarding the first `warmup_frac` of each replica's samples as extra warm-up.
"""
function read_rg(dir::AbstractString; warmup_frac::Real=0.1)
    rg = Float64[]
    for f in sort(readdir(dir))
        (startswith(f, "obs") && endswith(f, ".dat")) || continue
        vals = Float64[]
        for line in eachline(joinpath(dir, f))
            (isempty(line) || startswith(line, "#")) && continue
            cols = split(line)
            length(cols) >= 3 || continue
            push!(vals, parse(Float64, cols[3]))
        end
        n0 = floor(Int, warmup_frac * length(vals))
        append!(rg, @view vals[(n0 + 1):end])
    end
    return rg
end

"""
    pe_label(dir) -> Float64

Read the Pe value written by the sweep driver (`pe.txt`), falling back to
decoding the directory name (`pe_0p1` → 0.1).
"""
function pe_label(dir::AbstractString)
    pf = joinpath(dir, "pe.txt")
    isfile(pf) && return parse(Float64, strip(read(pf, String)))
    name = basename(dir)
    startswith(name, "pe_") || return NaN
    s = replace(name[4:end], "p" => ".", "m" => "-")
    return parse(Float64, s)
end

"""
    histogram(samples, edges) -> densities

Normalized density (integrates to 1) of `samples` over bin `edges`.
"""
function histogram(samples::Vector{Float64}, edges::AbstractVector{<:Real})
    nb = length(edges) - 1
    counts = zeros(Int, nb)
    for x in samples
        (x < edges[1] || x >= edges[end]) && continue
        b = searchsortedlast(edges, x)
        b = clamp(b, 1, nb)
        counts[b] += 1
    end
    w = diff(collect(edges))
    total = sum(samples .>= edges[1])
    dens = total == 0 ? zeros(nb) : counts ./ (total .* w)
    return dens
end

ascii_curve(dens, width=50) = begin
    m = maximum(dens; init=0.0)
    m == 0 && return fill(' ', length(dens))
    [repeat("█", round(Int, width * d / m)) for d in dens]
end

"""
    analyze(out_dir; nbins=40)

Read every `pe_*` subdirectory under `out_dir`, build P(R_G) per Pe on a
shared bin grid, and write:
  - `<out_dir>/pRg_vs_Pe.csv`  : bin centers + density column per Pe
  - `<out_dir>/mean_Rg_vs_Pe.csv` : Pe, mean, std, nsamples
An ASCII overlay and the summary table are printed. A PNG overlay is saved
if `Plots` can be loaded.
"""
function analyze(out_dir::AbstractString; nbins::Integer=40)
    subdirs = sort(filter(d -> isdir(joinpath(out_dir, d)) && startswith(d, "pe_"),
                          readdir(out_dir)))
    isempty(subdirs) && error("no pe_* subdirectories under $out_dir")

    pes = Float64[]
    samples = Vector{Float64}[]
    for d in subdirs
        full = joinpath(out_dir, d)
        rg = read_rg(full)
        isempty(rg) && (@warn "no R_G samples in $full"; continue)
        push!(pes, pe_label(full))
        push!(samples, rg)
    end
    order = sortperm(pes)
    pes, samples = pes[order], samples[order]

    # Shared bin grid across all Pe for a clean overlay.
    lo = minimum(minimum.(samples))
    hi = maximum(maximum.(samples))
    pad = 0.02 * (hi - lo + eps())
    edges = range(lo - pad, hi + pad; length=nbins + 1)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

    densities = [histogram(s, edges) for s in samples]

    # --- CSV: P(R_G) overlay -------------------------------------------
    pRg_path = joinpath(out_dir, "pRg_vs_Pe.csv")
    open(pRg_path, "w") do io
        println(io, "Rg," * join(("Pe=" * @sprintf("%g", pe) for pe in pes), ","))
        for i in eachindex(centers)
            @printf(io, "%.6g", centers[i])
            for d in densities
                @printf(io, ",%.6g", d[i])
            end
            print(io, '\n')
        end
    end

    # --- CSV: mean R_G vs Pe -------------------------------------------
    mean_path = joinpath(out_dir, "mean_Rg_vs_Pe.csv")
    open(mean_path, "w") do io
        println(io, "Pe,mean_Rg,std_Rg,nsamples")
        for (pe, s) in zip(pes, samples)
            @printf(io, "%g,%.6g,%.6g,%d\n", pe, mean(s), std(s), length(s))
        end
    end

    # --- Console summary ------------------------------------------------
    println("\n⟨R_G⟩ vs Pe  (N chain, pooled over replicas):")
    println(rpad("Pe", 10), rpad("⟨R_G⟩", 12), rpad("std", 12), "nsamples")
    for (pe, s) in zip(pes, samples)
        println(rpad(@sprintf("%g", pe), 10),
                rpad(@sprintf("%.4f", mean(s)), 12),
                rpad(@sprintf("%.4f", std(s)), 12),
                length(s))
    end

    println("\nP(R_G) overlay (each row a bin; bars scaled per-curve):")
    print(rpad("R_G", 8))
    for pe in pes
        print(rpad(@sprintf("Pe=%g", pe), 14))
    end
    println()
    curves = [ascii_curve(d, 12) for d in densities]
    for i in eachindex(centers)
        print(rpad(@sprintf("%.3f", centers[i]), 8))
        for c in curves
            print(rpad(c[i], 14))
        end
        println()
    end

    println("\nWrote:\n  ", pRg_path, "\n  ", mean_path)

    _try_plot(out_dir, centers, densities, pes)
    return (; pes, centers, densities)
end

# Optional PNG overlay if a plotting backend is installed. Never a hard dep.
function _try_plot(out_dir, centers, densities, pes)
    try
        @eval import Plots
        plt = Base.invokelatest(Plots.plot; xlabel="R_G", ylabel="P(R_G)",
                                title="P(R_G) vs Pe", legend=:topright)
        for (d, pe) in zip(densities, pes)
            Base.invokelatest(Plots.plot!, plt, centers, d;
                              label="Pe=" * @sprintf("%g", pe), lw=2)
        end
        png = joinpath(out_dir, "pRg_vs_Pe.png")
        Base.invokelatest(Plots.savefig, plt, png)
        println("  ", png)
    catch err
        @info "Plots backend unavailable; wrote CSV/ASCII only (install Plots for a PNG)." exception=(err,)
    end
    return nothing
end

end # module RgDistribution

# Allow standalone invocation.
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || (println(stderr, "usage: julia analysis/rg_distribution.jl <sweep_out_dir>"); exit(1))
    RgDistribution.analyze(ARGS[1])
end
