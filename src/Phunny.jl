"""
Extensions for making Sunny.jl usable
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
# Utility grids
# ---------------------------

ω_grid(ωmin, ωmax, n) = range(ωmin, ωmax; length=n)


# ---------------------------
# Validation & sanity checks
# ---------------------------

export validate_summary, asr_residual, rigid_translation_residual, rigid_rotation_residual,
       gamma_acoustic_energies, min_eigen_energy_along, sound_speeds



# Numerical ASR residual on Φ (should be ~0 after enforce_asr!)
function asr_residual(Φ::Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3,Float64,9}}, N::Int)
    maxnorm = 0.0
    for i in 1:N
        S = zeros(SMatrix{3,3,Float64,9})
        for ((ii,j,R), blk) in Φ
            ii == i || continue
            S += blk
        end
        maxnorm = max(maxnorm, opnorm(S, 2))
    end
    return maxnorm
end

# Rigid translation test: forces should vanish
function rigid_translation_residual(model::Model, Φ)
    N = model.N
    u = @SVector[1.0, 0.7, -0.4]     # arbitrary translation
    maxF = 0.0
    for i in 1:N 
	    S = zeros(SMatrix{3,3,Float64,9})
	    for ((ii,j,R), blk) in Φ
		    ii == i || continue
		    S += blk
	    end
	    Fi = S*u
	    maxF = max(maxF, norm(Fi))
    end
    return maxF
end

# Rigid rotation test: both net forces and torques should vanish
function rigid_rotation_residual(model::Model, Φ; θ::SVector{3,Float64}=@SVector[1e-3, -7e-4, 5e-4])
    N = model.N
    r = [model.lattice * model.fracpos[i] for i in 1:N]  # Å
    F = [zeros(SVector{3,Float64}) for _ in 1:N]

    for ((i,j,R), blk) in Φ
        rij = model.lattice * (model.fracpos[j] .+ SVector{3,Float64}(R) .- model.fracpos[i]) # Å
        Δu  = cross(θ, rij)
        f   = blk * Δu
        F[i] += f
        # equal-and-opposite applied to j would be -f (appears via (j,i,-R) entry)
    end

    Fnet = reduce(+, F)
    τnet = reduce(+, [cross(r[i], F[i]) for i in 1:N])
    return (norm(Fnet), norm(τnet), maximum(norm, F))
end

# Γ-point acoustic energies (meV) — should be ~0 (within tolerance)
function gamma_acoustic_energies(model::Model, Φ; cryst=nothing)
    EΓ, _ = phonons(model, Φ, @SVector[0.0,0.0,0.0]; q_basis=:cart, cryst=cryst)
    return sort!(EΓ[1:min(3, length(EΓ))])
end

# Minimum eigen-energy along a list of q (detect instabilities)
function min_eigen_energy_along(model::Model, Φ, qs::Vector{SVector{3,Float64}}; q_basis=:cart, q_cell=:primitive, cryst=nothing)
    minE = Inf
    arg  = nothing
    for q in qs
        E, _ = phonons(model, Φ, q; q_basis=q_basis, q_cell=q_cell, cryst=cryst)
        if E[1] < minE
            minE = E[1]; arg = q
        end
    end
    return (minE, arg)
end

# Sound speeds along given unit directions (Cartesian Å⁻¹) via finite differences near Γ
# Returns velocities for three acoustic branches (m/s) per direction.
function sound_speeds(model::Model, Φ, dirs::Vector{SVector{3,Float64}}; cryst=nothing, dq=1e-3)
    # Constants
    #ħ = ℏ                                 # J·s
    meV_to_J = 1.602176634e-22
    Åinv_to_minv = 1e10

    speeds = Vector{NTuple{3,Float64}}(undef, length(dirs))
    for (k, nhat) in pairs(dirs)
        q1 = dq * nhat                    # Å⁻¹
        E, _ = phonons(model, Φ, q1; q_basis=:cart, cryst=cryst)  # meV
        # Approximate slope dE/dq for the three acoustic branches
        dEdq = E[1:3] ./ dq               # meV·Å
        # Convert to m/s: v = (dE/dq)/ħ  with E in J, q in m⁻¹ → multiply by (meV→J) and (Å→m)
        v = ntuple(p-> (dEdq[p] * meV_to_J / ℏ) / Åinv_to_minv, 3)
        speeds[k] = v
    end
    return speeds
end

# High-level summary for quick realism checks
function validate_summary(model::Model, Φ; cryst=nothing, qpath_rlu=nothing, T=300.0)
    N = model.N
    out = Dict{Symbol,Any}()
    out[:ASR_residual] = asr_residual(Φ, N)
    out[:Rigid_translation_residual] = rigid_translation_residual(model, Φ)
    Fnet, τnet, Fmax = rigid_rotation_residual(model, Φ)
    out[:Rigid_rotation_force_net] = Fnet
    out[:Rigid_rotation_torque_net] = τnet
    out[:Rigid_rotation_force_max] = Fmax
    out[:Gamma_acoustic_meV] = gamma_acoustic_energies(model, Φ; cryst=cryst)

    # Optional path stability check in primitive RLU
    if qpath_rlu !== nothing
        qs = [q_cartesian(cryst, q; basis=:rlu, cell=:primitive) for q in qpath_rlu]
        minE, qarg = min_eigen_energy_along(model, Φ, qs; q_basis=:cart, cryst=cryst)
        out[:min_energy_on_path_meV] = minE
        out[:min_energy_q_cart] = qarg
    end

    # Sound speeds along axes
    dirs = [@SVector[1.0,0,0], @SVector[0,1.0,0], @SVector[0,0,1.0]]
    out[:sound_speeds_mps] = sound_speeds(model, Φ, dirs; cryst=cryst)

    return out
end



# ===========================
# Notes
# ===========================
# - Pairwise L/T decomposition K = kL*êêᵀ + kT*(I-êêᵀ) preserves translational invariance after ASR.
# - meV energy convention (ℏω): assumes k in eV/Å² and masses in amu; change ALPHA_meV if different.
# - q-space: `q_cartesian` interoperates with Sunny's primitive reciprocal lattice for RLU inputs.
# - Fast neighbor paths: use Sunny’s geometry and radius queries when available; otherwise portable cutoff.



end # module Phunny
