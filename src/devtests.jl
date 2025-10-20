#-------------------------------------------#
# List all available functions in Phunny.jl #
#-------------------------------------------#
list_functions() = filter(x->isa(getfield(Phunny, x), Function), names(Phunny; all=true))[21:end]
# ---------------------------#
# Validation & sanity checks #
# ---------------------------#
# Acoustic sum rule (ASR)
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

# Scheduled to be deprecated (theory check, dev)
# Γ-point acoustic energies (meV) — should be ~0 (within tolerance)
function gamma_acoustic_energies(model::Model, Φ; cryst=nothing)
    EΓ, _ = phonons(model, Φ, @SVector[0.0,0.0,0.0]; q_basis=:cart, cryst=cryst)
    return sort!(EΓ[1:min(3, length(EΓ))])
end
















