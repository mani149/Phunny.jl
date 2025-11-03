#=======================================================================================#
# This script defines all functions needed to perform numerical calculations using the  # 
# Phunny module. All necessary inputs are defined self-consistently or can be found in  #
# the constant.jl script. Optional functional features rely on helpers (helpers.jl).	#
#											#
# 	Section 1: Force Constant Matrices, Symmetry Constraints Dynamic Matrix, 	#
#		   Dynamic Gradient, & Phonon Eigenvalues/Eigenvectors			#
# 											#
#	Section 2: Mean-Squared Displacement, Anisotropic Debye-Waller Factor,   	#
#		   One-Phonon Dynamic Structure Factor [Path (2D) | Volume (4D)]	#
#											#
#											#
#							   Author(s): 			#
#								     `-> Isaac C. Ownby #
#=======================================================================================#
#----------------------------------------------------------# [ Section 1 ]
#                                                          #
#       Interatomic Force-Constant Matrix Assembly         # 
#                                                          #
#----------------------------------------------------------#
"""
    assemble_force_constants!(model)

Builds real-space force constants `Φ` as a dictionary of 3×3 blocks keyed by `(i,j,R)`, where `(i,j)`
are basis-atoms indices and `R` is a lattice vector in fractional coordinates. This function does
**not** mutate `model`.

Bond Stretching Block: 
For each harmonic bond `b` with equilibrium vector `r₀ = b.r0`, longitudinal/tangential stiffness
`(k_L, k_T) = (b.kL, b.kT)`, and unit direction `̂e = r₀/‖r0‖`; the force constant block is defined,
`K = kL*(êêᵀ) + kT*(I - êêᵀ)`, applied to the difference in displacements (uⱼ(R) - uᵢ(0)).

Blocks are accumulated:
    Φ[(i,j,+R)] += -K_b;  Φ[(j,i,-R)] += -K_b
    Φ[(i,i,0)]  += +K_b;  Φ[(j,j,0)]  += +K_b
ensuring Newton's third law. 

Types use StaticArrays: 
        keys → `Tuple{Int, Int, SVector{3,Int}}`
        vals → `SMatrix{3,3,Float64,9}`

Optional Bond-Angle (bending) Blocks:
If `β_bend > 0`, additional 3-body contributions are added for triplets `(i,j,k)`, where `j`
is the angle vertex and neighbors `(i,k)` are selected from same-cell bonds (`R==0`).
The parameter `bend_shell` determines the bonding scheme:
    - `:nn`  selects nearest-neighbor bonds with distance ≤ `(1 + bend_tol)*r_min`
    - `:all` selects all neighbors
else, a simple "second shell" cut-off is assumed. 

For each pair of neighbors `r_{ji}`, `r_{jk}` with norms `r_i` and `r_k` and directions
`̂e_i` and `̂e_k`, define two projectors `P_i = I - ̂e_i ̂e_iᵀ` and `P_k = I - ̂e_k ̂e_kᵀ`.

Then, the angular block matrices are defined:
    B_i  = P_i / r_i²;           B_k = P_k / r_k²
    R_ik = (P_i P_k)/(r_i r_k);  R_ki = R_ikᵀ
where each bending contribution scales with `β_bend` and is accumulated into on-site
and cross blocks among atoms `i, j` and `k` as:
    Φ[(i,i,0)] += β_bend*B_i
    Φ[(k,k,0)] += β_bend*B_k
    Φ[(j,j,0)] += β_bend*(B_i + B_k - R_ik - R_ki)

    Φ[(i,j,0)] += β_bend*(-B_i + R_ik);        Φ[(j,i,0)] += β_bend*(-B_i + R_ki)
    Φ[(j,k,0)] += β_bend*(-B_k + R_ki);        Φ[(k,j,0)] += β_bend*(-B_k + R_ik)
    Φ[(i,k,0)] += β_bend*(-R_ik);              Φ[(k,i,0)] += β_bend*(-R_ki)

Conservation laws are **not** enforced here and are enforced later by calling `enforce_asr!(Φ,model)`.


Returns
    - Φ::Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3,Float64,9}}

Notes
    - Units: `k_L`, `k_T`, and `β_bend` must be consistent with displacements in length units.
    - Symmetry: both `(i,j,R)` and `(j,i,−R)` are filled for pair terms; bending uses `R=0`.
"""
function assemble_force_constants!(model::Model; β_bend::Real=0.0, bend_shell::Symbol=:nn, bend_tol::Real=0.20)
Φ = Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3,Float64,9}}()
    I3 = SMatrix{3,3,Float64,9}(I)

    for b in model.bonds
        ê = b.r0 / norm(b.r0)
        PL = ê*ê'
        K  = b.kL*PL + b.kT*(I3 - PL)

        key_ij = (b.i, b.j, b.R)
        key_ji = (b.j, b.i, -b.R)
        key_ii = (b.i, b.i, SVector{3,Int}(0,0,0))
        key_jj = (b.j, b.j, SVector{3,Int}(0,0,0))

        Φ[key_ij] = get(Φ, key_ij, zeros(SMatrix{3,3,Float64,9})) - K
        Φ[key_ji] = get(Φ, key_ji, zeros(SMatrix{3,3,Float64,9})) - K
        Φ[key_ii] = get(Φ, key_ii, zeros(SMatrix{3,3,Float64,9})) + K
        Φ[key_jj] = get(Φ, key_jj, zeros(SMatrix{3,3,Float64,9})) + K
    end
    if β_bend > 0
        @inline function _add!(Φ, key, M)
            Φ[key] = get(Φ, key, zero(SMatrix{3,3,Float64,9})) + M
        end
        neighbor = Dict{Int, Vector{Phunny.Bond{Float64}}}()
        for b in model.bonds
            if b.R == SVector{3,Int}(0,0,0) #Same unit cell; generalize if desired
                push!(get(neighbor, b.i, Phunny.Bond{Float64}[]), b) #neighbors centered about `i`
                push!(get(neighbor, b.j, Phunny.Bond{Float64}[]), Phunny.Bond{Float64}(b.j,b.i,-b.R,-b.r0,b.kL,b.kT))
            end
        end

        for (j, nb) in neighbor
            L = length(nb); L < 2 && continue
            dists = [norm(b.r0) for b in nb]; rmin = minimum(dists)
            sel = if bend_shell == :nn
                [idx for (idx, d) in enumerate(dists) if d ≤ (1.0 + bend_tol)*rmin]
            elseif bend_shell == :all
                eachindex(nb)
            else 
                [idx for (idx, d) in enumerate(dists) if d > (1.0 + bend_tol*rmin)] #simple second shell
            end
            length(sel) < 2 && continue

            for a = 1:length(sel)-1, b = a+1:length(sel)
                
                ba = nb[sel[a]]; bb = nb[sel[b]]; i, k = ba.j, bb.j
                rji = ba.r0; rjk = bb.r0; ri = norm(rji); rk = norm(rjk)
                ei = rji/ri; ek = rjk/rk; Pi = I3 - ei*ei'; Pk = I3 - ek*ek'

                Bi = Pi/(ri^2); Bk = Pk/(rk^2) 
                Rik = (Pi*Pk)/(ri*rk); Rki = Rik'

                key_ii = (i,i,SVector{3,Int}(0,0,0)) 
                key_jj = (i,i,SVector{3,Int}(0,0,0)) 
                key_kk = (i,i,SVector{3,Int}(0,0,0))
                
                key_ij = (i,i,SVector{3,Int}(0,0,0)); key_ji = (i,i,SVector{3,Int}(0,0,0)) 
                key_jk = (i,i,SVector{3,Int}(0,0,0)); key_kj = (i,i,SVector{3,Int}(0,0,0))
                key_ik = (i,i,SVector{3,Int}(0,0,0)); key_ki = (i,i,SVector{3,Int}(0,0,0))
                
                _add!(Φ, key_ii, β_bend*Bi)
                _add!(Φ, key_kk, β_bend*Bk)
                _add!(Φ, key_jj, β_bend*(Bi + Bk - Rik - Rki))

                _add!(Φ, key_ij, β_bend*(-Bi + Rik))
                _add!(Φ, key_ji, β_bend*(-Bi + Rki))
                _add!(Φ, key_jk, β_bend*(-Bk + Rki))
                _add!(Φ, key_kj, β_bend*(-Bk + Rik))

                _add!(Φ, key_ik, β_bend*(-Rik))
                _add!(Φ, key_ki, β_bend*(-Rki))
            end
        end
    end
    return Φ
end
#-----------------------------------------------#
#               Acoustic Sum Rule               #
#                                               #
# Symmetry Constraint: Translational Invariance #
#-----------------------------------------------# 
"""
    enforce_asr!(Φ, N)

Acoustic sum rule: for each atom i, ∑_{j,R} Φ_{i j}(R) = 0.
Adjust on-site blocks to satisfy translational invariance.
"""
function enforce_asr!(Φ::Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3,Float64,9}}, N::Int)
    for i in 1:N
        sum_block = zeros(SMatrix{3,3,Float64,9})
        for ((ii,j,R), blk) in Φ
            ii == i || continue
            if !(j == i && R == SVector{3,Int}(0,0,0))
                sum_block += blk
            end
        end
        key_ii = (i,i,SVector{3,Int}(0,0,0))
        Φ[key_ii] = -sum_block
    end
    return Φ
end
#----------------------------------------#
#                                        #
#       Mass-weighted Dynamic Matrix     #
#                                        #
#----------------------------------------#
"""
    dynamical_matrix(model, Φ, q_cart)

Mass-weighted dynamical matrix D(q) from real-space Φ blocks.
`q_cart` is Cartesian (Å⁻¹). Returns Hermitian matrix.
"""
function dynamical_matrix(model::Model,
                          Φ::Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3,Float64,9}},
                          q_cart::SVector{3,Float64})
    N = model.N
    D = zeros(ComplexF64, 3N, 3N)
    M = model.mass

    for ((i,j,R), blk) in Φ
        r = model.lattice * (model.fracpos[j] .+ SVector{3,Float64}(R) .- model.fracpos[i])
        phase = cis(dot(q_cart, r))
        ii = (3(i-1)+1):(3i)
        jj = (3(j-1)+1):(3j)
        mw = blk ./ sqrt(M[i]*M[j])
        @inbounds D[ii, jj] .+= phase .* mw
    end

    D = (D + D')/2
    return Hermitian(D)
end
#--------------------------------#
# Mass-weighted Dynamic Gradient #
#--------------------------------#
"""
Fills pre-allocated dDx, dDy, dDz (3N×3N ComplexF64) with ∂D/∂q components.
Arrays are overwritten; no allocation in the hot loop beyond scalars.
"""
function dynamical_gradient!(dDx::AbstractMatrix{ComplexF64},
                             dDy::AbstractMatrix{ComplexF64},
                             dDz::AbstractMatrix{ComplexF64},
                             model::Model,
                             Φ::Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3,Float64,9}},
                             q_cart::SVector{3,Float64})
    N = model.N
    M = model.mass
    fill!(dDx, 0); fill!(dDy, 0); fill!(dDz, 0)

    @inbounds for ((i,j,R), blk) in Φ
        r = model.lattice * (model.fracpos[j] .+ SVector{3,Float64}(R) .- model.fracpos[i])
        phase = cis(dot(q_cart, r))
        ii = (3(i-1)+1):(3i)
        jj = (3(j-1)+1):(3j)
        mw = blk ./ sqrt(M[i]*M[j])

        facx = (im * r[1]) * phase
        facy = (im * r[2]) * phase
        facz = (im * r[3]) * phase

        dDx[ii, jj] .+= facx .* mw
        dDy[ii, jj] .+= facy .* mw
        dDz[ii, jj] .+= facz .* mw
    end

    # symmetrize
    dDx .= (dDx .+ dDx') ./ 2
    dDy .= (dDy .+ dDy') ./ 2
    dDz .= (dDz .+ dDz') ./ 2
    return nothing
end
#-------------------------------------------#
# Mass-Weighted Directional Dynamic Hessian #
#-------------------------------------------#
"""
dynamical_hessian!(Hn, model, Φ, nhat; backend=:analytic, h=1e-2)

Fill Hn with the directional Hessian D^{(2)}_{n̂} at Γ (INTERNAL units,
same mass-weighted basis/units as `dynamical_matrix`).

Backends:
  :analytic     -- exact in harmonic: one sweep over Φ with factor -(n·r)^2
  :complexstep  -- high-accuracy numeric using D(ih n) without API changes
  :FiniteDiff   -- central finite difference of ∂D/∂q along n
  :AutoDiffHVP  -- not implemented in this draft (placeholder)

Keyword:
  h::Float64 = 1e-2  -- step size; for :complexstep you can go much smaller
"""
function dynamical_hessian!(Hn::AbstractMatrix{ComplexF64},
                            model::Model,
                            Φ::Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3,Float64,9}},
                            nhat::SVector{3,Float64};
                            backend::Symbol = :analytic,
                            h::Float64 = 1e-2)

    N  = model.N
    M  = model.mass
    nh = nhat / norm(nhat)
    fill!(Hn, 0)

    if backend === :analytic
        @inbounds for ((i,j,R), blk) in Φ
            r   = model.lattice * (model.fracpos[j] .+ SVector{3,Float64}(R) .- model.fracpos[i])
            fac = - (dot(nh, r))^2
            ii  = (3(i-1)+1):(3i)
            jj  = (3(j-1)+1):(3j)
            mw  = blk ./ sqrt(M[i]*M[j])
            Hn[ii, jj] .+= fac .* mw
        end

    elseif backend === :complexstep
        # We cannot call Phunny.dynamical_matrix with complex q (its signature is Float64),
        # so we reproduce its loop here to evaluate D(q) at q = 0 and q = i*h*nh safely.
        # Identity: D(ih n) = D(0) + i h D' - (h^2/2) D'' + ...
        #  => D'' ≈ 2*( D(0) - Re[D(ih n)] ) / h^2
        N3 = 3N
        D0  = zeros(ComplexF64, N3, N3)
        Dim = zeros(ComplexF64, N3, N3)  # D(i h n)

        @inbounds for ((i,j,R), blk) in Φ
            r      = model.lattice * (model.fracpos[j] .+ SVector{3,Float64}(R) .- model.fracpos[i])
            ii     = (3(i-1)+1):(3i)
            jj     = (3(j-1)+1):(3j)
            mw     = blk ./ sqrt(M[i]*M[j])
            # q = 0 → phase = 1
            D0[ii, jj]  .+= mw
            # q = i h n → phase = exp(i * (i h n·r)) = exp(-h * n·r) (real, positive)
            th         = -h * dot(nh, r)
            phase_im   = exp(th)           # real scalar
            Dim[ii, jj] .+= phase_im .* mw
        end

        # Symmetrize D0 and Dim (Hermitian numerically)
        D0  .= (D0  .+ D0') ./ 2
        Dim .= (Dim .+ Dim') ./ 2

        # Directional Hessian via complex-step identity (no cancellation)
        Hn .= 2 .* (D0 .- real.(Dim)) ./ (h^2)

    elseif backend === :FiniteDiff
        # D''_n ≈ ( D'_n(+h) - D'_n(-h) ) / (2h), where D'_n = n·∇D
        N3   = 3N
        dDxp = zeros(ComplexF64, N3, N3); dDyp = similar(dDxp); dDzp = similar(dDxp)
        dDxm = zeros(ComplexF64, N3, N3); dDym = similar(dDxm); dDzm = similar(dDxm)
        qp   = h .* nh
        qm   = -h .* nh
        dynamical_gradient!(dDxp, dDyp, dDzp, model, Φ, qp)
        dynamical_gradient!(dDxm, dDym, dDzm, model, Φ, qm)
        @. Hn = ( nh[1]*(dDxp - dDxm) + nh[2]*(dDyp - dDym) + nh[3]*(dDzp - dDzm) ) / (2h)

    elseif backend === :AutoDiffHVP
        error("AutoDiffHVP backend not implemented in this draft. Try backend=:analytic or :complexstep.")

    else
        error("Unknown backend = $backend. Use one of: :analytic, :complexstep, :FiniteDiff, :AutoDiffHVP.")
    end

    # Numeric Hermitian symmetrization
    Hn .= (Hn .+ Hn') ./ 2
    return Hn
end
#--------------------------------------------------------#
#                                                        #
#        Phonon Energy Eigenvalues & Eigenvectors        #
#                                                        #
#--------------------------------------------------------#
"""
    phonons(model, Φ, q; q_basis=:cart, q_cell=:primitive, cryst=nothing)

Diagonalizes the mass-weighted dynamic matrix D(q). Returns (eigvals, eigvecs) with **energies in meV** and a 3N×3N 
block matrix with columns defined by the eigenvectors (phonon polarization vectors).
The wavevector `q` may be in Cartesian Å⁻¹ (`q_basis=:cart`) or relative lattice untis (R.L.U.) (`q_basis=:rlu` with `q_cell=:primitive` or `:conventional`).
If passing RLU, also pass the underlying crystal `cryst` via keyword.
"""
function phonons(model::Model, Φ, q::SVector{3,Float64}; q_basis::Symbol=:cart, q_cell::Symbol=:primitive, cryst=nothing)
    q_cart = q_basis === :cart ? q : (cryst === nothing ? error("cryst required for q_basis=:rlu") : q_cartesian(cryst, q; basis=:rlu, cell=q_cell))
    Dq = dynamical_matrix(model, Φ, q_cart)
    vals, vecs = eigen(Dq)
    ω2 = max.(real(vals), 0.0)
    ω  = sqrt.(ω2)                    # internal sqrt(k/m) units
    E  = ALPHA_meV .* ω               # energies in meV (ℏω)
    perm = sortperm(E)
    return E[perm], vecs[:,perm]
end
#========================================================================#
#--------------------------------------------------# [ Section 2 ]
#                                                  #
#      Mean-square displacements from phonons      #
#                                                  #
#--------------------------------------------------#
"""
    msd_from_phonons(model, Φ; T, cryst, qgrid=(12,12,12), q_cell=:primitive, eps_meV=1e-6)

Computes the mean-square displacements ⟨u_s^2⟩(T) per-site in Å² directly from the phonon spectrum:
⟨u_s^2⟩ = (1/Nq) * Σ_{q,ν} (ħ² / (2 M_s E_J)) * coth(E_J / (2 k_B T)) * Σ_α |e_phys(s,α;q,ν)|²
where E_J = E_meV ⋅ meV_to_J and e_phys are physical (non mass-weighted) polarization vectors.
Modes with E ≤ eps_meV are skipped (Γ handling); Convergence may require refinement of qgrid.
"""
function msd_from_phonons(model::Model, Φ;
                          T::Real, cryst, qgrid::NTuple{3,Int}=(12,12,12),
                          q_cell::Symbol=:primitive, eps_meV::Real=5e-2)
    N = model.N
    nx, ny, nz = qgrid
    Nq = nx*ny*nz
    msd_A2 = zeros(Float64, N)   # accumulate in Å^2

    for iz in 0:nz-1, iy in 0:ny-1, ix in 0:nx-1
        q_rlu = @SVector[ix/nx, iy/ny, iz/nz]
        Eν, Evec = phonons(model, Φ, q_rlu; q_basis=:rlu, q_cell=q_cell, cryst=cryst)
        ν_start = (ix==0 && iy==0 && iz==0) ? 4 : 1
        for ν in ν_start:length(Eν)
            EmeV = Eν[ν]
            EmeV <= eps_meV && continue
            x = EmeV / (2*K_B_meV_per_K*T)
            cothx = 1.0 / tanh(x)
            for s in 1:N
                i1 = 3s - 2; i2 = 3s - 1; i3 = 3s
                # mass-weighted eigenvector components
                e1 = Evec[i1, ν]; e2 = Evec[i2, ν]; e3 = Evec[i3, ν]
                amp2 = (abs2(e1) + abs2(e2) + abs2(e3)) / model.mass[s]  # |e_phys|^2 sum
                msd_A2[s] += MSD_PREF_A2 * cothx * amp2 / EmeV
            end
        end
    end

    msd_A2 ./= Nq
    return msd_A2
end
#-----------------------------------------------------------------------#
#                                                                       #
# Full anisotropic displacement tensors U^{(s)}_{αβ} from phonons (Å^2) #
#                                                                       #
#-----------------------------------------------------------------------#
"""
   U_from_phonons(model, Φ; T, cryst, qgrid=(12,12,12), q_cell=:primitive, eps_meV=1e-6)

Returns the full anisotropic displacement tensors U^{(s)}_{αβ} from the output of phonons(model,Φ).

"""
function U_from_phonons(model::Model, Φ;
                        T::Real, cryst, qgrid::NTuple{3,Int}=(12,12,12),
                        q_cell::Symbol=:primitive, eps_meV::Real=2e-1)
    N = model.N
    nx, ny, nz = qgrid
    Nq = nx*ny*nz
    U = [zeros(SMatrix{3,3,Float64,9}) for _ in 1:N]

    for iz in 0:nz-1, iy in 0:ny-1, ix in 0:nx-1
        q_rlu = @SVector[(ix+0.5)/nx, (iy+0.5)/ny, (iz+0.5)/nz]
        Eν, Evec = phonons(model, Φ, q_rlu; q_basis=:rlu, q_cell=q_cell, cryst=cryst)
        ν_start = (ix==0 && iy==0 && iz==0) ? 4 : 1
        for ν in ν_start:length(Eν)
            EmeV = Eν[ν]
            EmeV <= eps_meV && continue
            x = EmeV / (2*K_B_meV_per_K*T)
            cothx = 1.0 / tanh(x)
            for s in 1:N
                i1 = 3s - 2; i2 = 3s - 1; i3 = 3s
                e1 = Evec[i1, ν]; e2 = Evec[i2, ν]; e3 = Evec[i3, ν]
                # mass-weighted polarization tensor components: (eα eβ*)
                t11 = (e1*conj(e1)); t12 = (e1*conj(e2)); t13 = (e1*conj(e3))
                t21 = (e2*conj(e1)); t22 = (e2*conj(e2)); t23 = (e2*conj(e3))
                t31 = (e3*conj(e1)); t32 = (e3*conj(e2)); t33 = (e3*conj(e3))
                Tmat = @SMatrix[t11 t12 t13; t21 t22 t23; t31 t32 t33] ./ model.mass[s]
                U[s] += (MSD_PREF_A2 * cothx / EmeV) * real.(Tmat)
            end
        end
    end
    for s in 1:N
        U[s] = U[s] ./ Nq
	U[s] = (U[s] + U[s]')/2
    end
    return U
end
#---------------------------------#
# Internal helpers for one-phonon #
# coherent DSF calculations       #
#---------------------------------#
# simple erf approximation (Abramowitz–Stegun, good to 1e-7)
@inline function _erf(x::Float64)
    # Abramowitz–Stegun 7.1.26
    a1=0.254829592; a2=-0.284496736; a3=1.421413741
    a4=-1.453152027; a5=1.061405429; p=0.3275911
    s = signbit(x) ? -1.0 : 1.0
    x = abs(x)
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t)*exp(-x*x)
    return s*y
end
#resolve coherent scattering lengths (fm) to length-N vector
@inline function _resolve_bcoh(
    model::Model;
    bcoh=nothing,
    iso_by_site::Union{Nothing,Dict{Int,Int}}=nothing,
    iso_by_species::Union{Nothing,Dict{Symbol,Int}}=nothing,
)::Vector{Float64}
    N = model.N
    if bcoh === nothing
        # Coerce to concrete Dicts to match your bcoh_lookup signature
        iso_site   = (iso_by_site    === nothing) ? Dict{Int,Int}()        : iso_by_site
        iso_species= (iso_by_species === nothing) ? Dict{Symbol,Int}()     : iso_by_species
        try
            return bcoh_lookup(model.species; iso_by_site=iso_site, iso_by_species=iso_species)
        catch err
            @warn "bcoh auto-lookup failed; defaulting to ones(N). Error: $err"
            return ones(Float64, N)
        end
    elseif bcoh isa Number
        return fill(Float64(bcoh), N)
    else
        length(bcoh) == N || error("bcoh must be a scalar or a length-N vector (N=$(N)).")
        return Float64.(bcoh)
    end
end
#-------------------------------------------------------#
#   One-phonon Coherent Dynamic Structure Factor (meV)  #
#                                                       #
# q-path dataset ~ onephonon_dsf                        #
# 4-dim  dataset ~ onephonon_dsf_4d                     #
#                                                       #
#-------------------------------------------------------#
"""
    onephonon_dsf(model, Φ, q, Evals; T=300.0, bcoh=nothing, η=0.5, mass_unit=:amu, q_basis=:cart, q_cell=:primitive, cryst=nothing)

Computes one-phonon coherent S(q,E) on energy grid `Evals` (meV).
- Debye–Waller is intrinsic: full anisotropic tensors U_s(T) are computed from the phonon spectrum.
- `bcoh` optional Vector of coherent scattering lengths (length N)
- `η` Gaussian HWHM for energy broadening (meV)
- `mass_unit` :amu or :kg (Model.mass is assumed to be :amu typically)
- `q_basis`/`q_cell`/`cryst` follow `phonons` conventions
Returns S(E) (arbitrary but consistent units).
"""
function onephonon_dsf(model::Model, Φ, q::SVector{3,Float64}, Evals::AbstractVector{<:Real};
                        T::Real=300.0, bcoh=nothing, η::Real=0.5, mass_unit::Symbol=:amu,
                        q_basis::Symbol=:cart, q_cell::Symbol=:primitive, cryst=nothing,
                        dw_qgrid::NTuple{3,Int}=(16,16,16), _U_internal::Union{Nothing,Vector{SMatrix{3,3,Float64,9}}}=nothing,
			iso_by_site::Union{Nothing,Dict{Int,Int}}=nothing, iso_by_species::Union{Nothing,Dict{Symbol,Int}}=nothing)

    #Number of particles per unit cell & mass
    N = model.N; M = model.mass
    #Resolve mass units
    Mamu = mass_unit === :amu ? M : mass_unit === :kg ? (M ./ u_to_kg) : error("mass_unit must be :amu or :kg")

    #Automatic lookup for coherent scattering length
    bw = _resolve_bcoh(model;bcoh=bcoh, iso_by_site=iso_by_site, iso_by_species=iso_by_species)

    #Resolve momentum/position units & precompile phase info
    qvec = q_basis === :cart ? q : q_cartesian(cryst, q; basis=:rlu, cell=q_cell)
    rvec = q_basis === :cart ? [model.lattice*model.fracpos[s] for s in 1:N] : [model.fracpos[s] for s in 1:N]
    phase = q_basis === :cart ? [exp(im*dot(qvec, rvec[s])) for s in 1:N] : [exp(im*dot(2π*qvec, rvec[s])) for s in 1:N]
    
    #Calculate anisotropic Debye-Waller
    U = _U_internal === nothing ? U_from_phonons(model, Φ; T=T, cryst=cryst, qgrid=dw_qgrid, q_cell=q_cell) : _U_internal
    DW = [exp(-0.5*dot(qvec, U[s]*qvec)) for s in 1:N]

    # Solve eigenvalue problem; Energies in meV from phonons(); eigenvectors are mass-weighted
    Eν, Evec = phonons(model, Φ, q; q_basis=q_basis, q_cell=q_cell, cryst=cryst)
    #Assert that Emin = -1*Emax or 0 (until arbitrary grid normalization is implemented)  
    Emin, Emax = extrema(Evals); @assert Emin == -Emax || Emin == 0.0 "Minimum Energy MUST be zero or -1*Emax!"
    
    # Bose factor in meV units
    nB = @. 1/(exp(Eν/(K_B_meV_per_K*T)) - 1)

    #Dynamic Structure Factor
    Sω = zeros(Float64, length(Evals))
    for ν in 1:length(Eν)
        EmeV = Eν[ν]
        EmeV <= 0 && continue
        
        #Coherent scattering amplitude
        Aq = 0.0 + 0.0im
        for s in 1:N
            i1 = 3s - 2; i2 = 3s - 1; i3 = 3s
            # physical polarization
	    e1 = Evec[i1, ν]; e2 = Evec[i2, ν]; e3 = Evec[i3, ν]
            qdot = qvec[1]*e1 + qvec[2]*e2 + qvec[3]*e3
            Aq += ( (bw[s] * DW[s] * qdot) / sqrt(2 * Mamu[s] * EmeV) )*phase[s]
        end
        #Squared Amplitude (Intensity)
        Iq = abs2(Aq)

        # Fraction of +ω Gaussian captured
        ωfracp = 0.5 * (_erf((Emax - EmeV) / (sqrt(2)*η)) - _erf((Emin - EmeV) / (sqrt(2)*η)))
        # Fraction of -ω Gaussian captured
        ωfracm = 0.5 * (_erf((Emax + EmeV) / (sqrt(2)*η)) - _erf((Emin + EmeV) / (sqrt(2)*η)))

        #Calculate scattering intensity
        ωfracp > 1e-12 || continue; np = nB[ν] + 1.0; nm = nB[ν]
        @inbounds @simd for k in eachindex(Evals)
            E = Evals[k]
            Sω[k] += Iq * np * exp(-((E - EmeV)^2)/(2η^2)) / (sqrt(2π)*η) / ωfracp
            if Emin < 0.0 && ωfracm > 1e-12
                Sω[k] += Iq * nm * exp(-((E + EmeV)^2)/(2η^2)) / (sqrt(2π)*η) / ωfracm
            end
        end
    end
    return Sω
end
#-----------------------------------------#
# 4D S(q, w) over a Cartesian or RLU grid #
#-----------------------------------------#
"""
    onephonon_dsf_4d(model, Φ, q1, q2, q3, Evals;
                     q_basis=:rlu, q_cell=:primitive, cryst=nothing,
                     T=300.0, η=0.5, mass_unit=:amu, bcoh=nothing,
                     threads=true)

Compute the one-phonon coherent dynamic structure factor on a **4D grid**:

- `q1, q2, q3` are 1D axes defining a rectilinear grid in q-space.
  - If `q_basis=:rlu`, they are fractional coordinates along the three reciprocal basis vectors.
  - If `q_basis=:cart`, they are Cartesian components (Å⁻¹).
- `Evals` is the 1D energy axis (meV). Result has shape `(length(q1), length(q2), length(q3), length(Evals))`.

This routine is thread-parallel over the q-grid to mitigate the O(N^3) complexity.
"""
function onephonon_dsf_4d(model::Model, Φ,
                          q1::AbstractVector, q2::AbstractVector, q3::AbstractVector,
                          Evals::AbstractVector;
                          q_basis::Symbol=:rlu, q_cell::Symbol=:primitive, cryst=nothing,
                          T::Real=300.0, η::Real=0.5, mass_unit::Symbol=:amu, bcoh=nothing,
                          threads::Bool=true, dw_qgrid::NTuple{3,Int}=(12,12,12),
			  iso_by_site::Union{Nothing,Dict{Int,Int}}=nothing, 
			  iso_by_species::Union{Nothing,Dict{Symbol,Int}}=nothing)

    n1, n2, n3, nE = length(q1), length(q2), length(q3), length(Evals)
    S4 = zeros(Float64, n1, n2, n3, nE)

    # Precompute anisotropic Debye–Waller tensors once (Å^2)
    U = U_from_phonons(model, Φ; T=T, cryst=cryst, qgrid=dw_qgrid, q_cell=q_cell)

    #NEW: resolve bcoh once (thread-safe, read-only, vector) 
    #Note: onephonon_dsf(...; bcoh=bw) was onephonon_dsf(...; bcoh=bcoh) before automatic look-up
    bw = _resolve_bcoh(model; bcoh=bcoh, iso_by_site=iso_by_site, iso_by_species=iso_by_species)

    # Prepare reciprocal matrix for RLU if needed
    G = nothing
    if q_basis === :rlu
        cryst === nothing && error("cryst must be provided when q_basis=:rlu")
        if isdefined(Main, :Sunny) && q_cell === :primitive
            S = getfield(Main, :Sunny)
            G = S.prim_recipvecs(cryst)     # 3x3 (Å⁻¹)
        else
            L = Matrix(getproperty(cryst, :latvecs))
            G = 2π * inv(L)'                # 3x3 (Å⁻¹)
        end
    end

    # Inner worker: compute at a single (i,j,k)
    function _compute_at!(i, j, k)
        h = q1[i]; kk = q2[j]; l = q3[k]
        qcart = if q_basis === :cart
            @SVector[h, kk, l]
        else
            v = G * SVector{3,Float64}(h,kk,l)
            SVector{3,Float64}(v)
        end
        Sq = onephonon_dsf(model, Φ, qcart, Evals; T=T, bcoh=bw,
                           η=η, mass_unit=mass_unit, q_basis=:cart, cryst=cryst,
                           dw_qgrid=dw_qgrid, _U_internal=U)
        @inbounds S4[i, j, k, :] .= Sq
        return nothing
    end

    if threads
        Base.Threads.@threads for t in 1:(n1*n2*n3)
            i = ((t-1) % n1) + 1
            j = ((t-1) ÷ n1) % n2 + 1
            k = ((t-1) ÷ (n1*n2)) % n3 + 1
            _compute_at!(i, j, k)
        end
    else
        for k in 1:n3, j in 1:n2, i in 1:n1
            _compute_at!(i, j, k)
        end
    end

    return S4
end








