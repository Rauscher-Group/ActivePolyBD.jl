# I/O (§7): XYZ trajectory, scalar observable log, and checkpoint/restart.
# All frequencies are caller-controlled; writes are buffered and flushed.

using Serialization

# Element labels so terminal (passive) beads are visible in VMD/OVITO.
const ELEM_END = "O"       # passive terminal monomers (paper colors these red)
const ELEM_INTERIOR = "C"  # active interior monomers

"""
    element_label(i, N) -> String

`ELEM_END` for the two terminal monomers, `ELEM_INTERIOR` otherwise.
"""
@inline element_label(i::Int, N::Int) = (i == 1 || i == N) ? ELEM_END : ELEM_INTERIOR

# --- XYZ trajectory ------------------------------------------------------

"""
    write_xyz_frame!(io, sys, step, time, Rg)

Append one extended-XYZ frame: count, a `step time Rg` comment, then
`element x y z` per monomer.
"""
function write_xyz_frame!(io::IO, sys::System, step::Integer, time::Real, Rg::Real)
    N = sys.N
    println(io, N)
    @printf(io, "step=%d time=%.6g Rg=%.6g\n", step, time, Rg)
    @inbounds for i in 1:N
        r = sys.positions[i]
        @printf(io, "%s %.8g %.8g %.8g\n", element_label(i, N), r[1], r[2], r[3])
    end
    return io
end

# --- Scalar observable log ----------------------------------------------

"""
    write_log_header(io; extra_cols=String[])

Comment header for the scalar `.dat` log. Base columns are
`step time Rg cm_x cm_y cm_z`; `extra_cols` appends optional column names.
"""
function write_log_header(io::IO; extra_cols::Vector{String}=String[])
    cols = ["step", "time", "Rg", "cm_x", "cm_y", "cm_z"]
    append!(cols, extra_cols)
    println(io, "# ", join(cols, " "))
    return io
end

"""
    write_log_row!(io, step, time, Rg, cm; extra=Float64[])

One whitespace-delimited data row. `extra` appends optional column values.
"""
function write_log_row!(io::IO, step::Integer, time::Real, Rg::Real, cm::Vec3;
                        extra::Vector{Float64}=Float64[])
    @printf(io, "%d %.8g %.8g %.8g %.8g %.8g", step, time, Rg, cm[1], cm[2], cm[3])
    for v in extra
        @printf(io, " %.8g", v)
    end
    print(io, '\n')
    return io
end

# --- Checkpoint / restart ------------------------------------------------

struct Checkpoint
    positions::Vector{Vec3}
    step::Int
    rng
end

"""
    save_checkpoint(path, sys, step, rng)

Serialize positions, the current production step, and the full RNG state so
a run can resume bit-for-bit. Uses the `Serialization` stdlib (no extra dep).
"""
function save_checkpoint(path::AbstractString, sys::System, step::Integer, rng)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, Checkpoint(copy(sys.positions), Int(step), deepcopy(rng)))
    end
    mv(tmp, path; force=true)   # atomic-ish replace so a crash can't truncate
    return path
end

"""
    load_checkpoint(path) -> Checkpoint

Read a checkpoint written by [`save_checkpoint`](@ref).
"""
function load_checkpoint(path::AbstractString)
    return open(deserialize, path, "r")::Checkpoint
end

"""
    restart!(sys, path) -> (step, rng)

Load `path` into `sys.positions` in place and return the saved step and RNG
so the caller can continue the production loop.
"""
function restart!(sys::System, path::AbstractString)
    cp = load_checkpoint(path)
    length(cp.positions) == sys.N ||
        error("checkpoint has $(length(cp.positions)) monomers, system has $(sys.N)")
    copyto!(sys.positions, cp.positions)
    return cp.step, cp.rng
end
