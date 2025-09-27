## Should be included in 'helpers.jl' for future
## future parent of 'calculations.jl'

 ## per-system
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


## per-system
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

## per-system
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
