"""
A semi-classical approach to solve for phonon frequencies and eigenmodes, built for integration with Sunny.jl.
"""
module Phunny

using LinearAlgebra, SparseArrays, StaticArrays

# ===========================
# Public API
# ===========================

export Model, Bond, build_model, neighbor_bonds_cutoff, neighbor_bonds_from_sunny,
       assemble_force_constants!, enforce_asr!, dynamical_matrix, phonons,
       onephonon_dsf, ω_grid, mass_vector, q_cartesian, onephonon_dsf_4d, collapse,
       mass_lookup, bcoh_lookup, msd_from_phonons, B_isotropic_from_phonons, U_from_phonons


include("constants.jl")
include("helpers.jl")
include("calculations.jl")

# ---------------------------
# Validation & sanity checks
# ---------------------------

export validate_summary, asr_residual, rigid_translation_residual, rigid_rotation_residual,
       gamma_acoustic_energies, min_eigen_energy_along, sound_speeds
       
include("devtests.jl")
include("systests.jl")


end # module Phunny
