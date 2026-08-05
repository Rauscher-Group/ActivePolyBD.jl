"""
    ActivePolyBD

Overdamped Brownian-dynamics simulator for tangentially active bead–spring
polymers (α = 0), in reduced units, following Bianco, Locatelli & Malgaretti,
PRL 121, 217802 (2018). See `ACTIVE_POLYMER_IMPLEMENTATION_PLAN.md` for the
normative physics spec.

Reduced units: b = 1, k_BT = 1, τ_B = 1 ⇒ D₀ = βD₀ = 1, and the reduced
active-force magnitude equals the Péclet number (f^act = Pe).
"""
module ActivePolyBD

using StaticArrays
using LinearAlgebra
using Random
using Printf
using Statistics
using TOML

# Abstract types + concrete potentials/activity/neighbors come first so the
# concretely-parameterized `System` in types.jl can reference them.
include("potentials_bonded.jl")
include("potentials_pair.jl")
include("activity.jl")
include("neighbors.jl")
include("types.jl")
include("forces.jl")
include("observables.jl")
include("integrator.jl")
include("init.jl")
include("io.jl")
include("config.jl")
include("simulation.jl")

# Types
export System, SimParams, RunConfig, OutputSpec, deepcopy_system
export BondedPotential, HarmonicBond, FENEBond
export PairPotential, SoftRepulsive, WCA
export ActivityModel, NoActivity, TangentialActivity
export NeighborStrategy, AllPairs, VerletList

# Physics
export dVdr, energy, cutoff, coupling, with_coupling
export compute_forces!, bd_step!

# Observables
export center_of_mass, gyration_tensor, radius_of_gyration, radius_of_gyration_sq
export gyration_eigenvalues, end_to_end, end_to_end_sq, backbone_cosangles
export AsphericityAccumulator, asphericity
export bonded_energy, pair_energy, nonbonded_energy, nonbonded_energies

# Init / IO
export initialize_chain
export write_xyz_frame!, write_log_header, write_log_row!
export save_checkpoint, load_checkpoint, restart!

# Config / driver
export load_config, parse_config, build_system
export run_single!, run!, ReplicaResult, replica_rng, suffixed

end # module ActivePolyBD
