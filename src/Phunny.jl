"""
Isaac's extension for making Sunny.jl usable
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

# ---------------------------
# Data structures
# ---------------------------

struct Bond{T}
    i::Int                     # atom index in home cell (1..N)
    j::Int                     # atom index in cell displaced by R
    R::SVector{3,Int}          # lattice offset (in integer supercell coordinates)
    r0::SVector{3,Float64}     # equilibrium cartesian bond vector from i@0 to j@R (Å)
    kL::T                      # longitudinal spring (eV/Å^2 by convention)
    kT::T                      # transverse spring   (eV/Å^2 by convention)
end

mutable struct Model{T}
    lattice::SMatrix{3,3,Float64,9}         # columns = a1 a2 a3 (Å)
    fracpos::Vector{SVector{3,Float64}}     # fractional positions in [0,1)
    species::Vector{Symbol}                 # species labels
    mass::Vector{T}                         # length N, mass units (amu or kg; see DSF kwarg)
    bonds::Vector{Bond{T}}                  # pair list with (kL,kT)
    N::Int
end
# ---- internal accumulation helper ----
@inline function _add_block!(Φ::Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3,Float64,9}},
                             key::Tuple{Int,Int,SVector{3,Int}}, M::SMatrix{3,3,Float64,9})
    Φ[key] = get(Φ, key, zeros(SMatrix{3,3,Float64,9})) + M
    return nothing
end


# Convenience
mass_vector(m::Model) = m.mass

# ---------------------------
# Helpers
# ---------------------------

@inline frac_to_cart(L::SMatrix{3,3,Float64,9}, f::SVector{3,Float64}) = L * f

# Access Sunny module lazily (optional dependency)
@inline function _sunny_mod()
    isdefined(Main, :Sunny) || error("Sunny.jl must be loaded to use this feature")
    return getfield(Main, :Sunny)
end

# ---------------------------
# q-space conversion
# ---------------------------

"""
    q_cartesian(cryst, q; basis=:cart, cell=:primitive)

Convert a wavevector to Cartesian (Å⁻¹). `q` may be:
- `basis=:cart`    → already Cartesian (Å⁻¹)
- `basis=:rlu`     → reduced lattice units (h,k,l). `cell` chooses `:primitive` or `:conventional`.
Uses `Sunny.prim_recipvecs(cryst)` if Sunny is loaded and `cell=:primitive`, otherwise falls back to
`2π * inv(L)'` where `L` is the corresponding direct lattice.
"""
function q_cartesian(cryst, q::SVector{3,Float64}; basis::Symbol=:cart, cell::Symbol=:primitive)
    if basis === :cart
        return q
    elseif basis === :rlu
        if cell === :primitive && isdefined(Main, :Sunny)
            S = _sunny_mod()
            G = S.prim_recipvecs(cryst)             # 3x3 columns b1,b2,b3 (Å⁻¹)
            return SVector{3,Float64}(G * q)
        else
            # conventional cell
            L = SMatrix{3,3,Float64,9}(Matrix(getproperty(cryst, :latvecs)))
            G = 2π * inv(L)'                         # columns b1,b2,b3
            return SVector{3,Float64}(G * q)
        end
    else
        throw(ArgumentError("basis must be :cart or :rlu"))
    end
end

# ---------------------------
# Neighbor/bond construction
# ---------------------------

"""
    neighbor_bonds_cutoff(lattice, fracpos; cutoff, kL, kT, supercell=(1,1,1))

Construct bonds using a radial cutoff by scanning a small supercell (portable path).
`kL` and `kT` can be numbers or functions `(i,j,rij)->stiffness` for non-uniform springs.
Returns `Vector{Bond}`.
"""
function neighbor_bonds_cutoff(lattice::SMatrix{3,3,Float64,9},
                               fracpos::Vector{SVector{3,Float64}};
                               cutoff::Real,
                               kL::Union{Real,Function}=1.0,
                               kT::Union{Real,Function}=1.0,
                               supercell::NTuple{3,Int}=(1,1,1))
    S = SVector{3,Int}(supercell)
    N = length(fracpos)
    bonds = Bond{Float64}[]
    cutoff2 = cutoff^2

    for i in 1:N
        ri = frac_to_cart(lattice, fracpos[i])
        for j in 1:N
            for Rx in -(S[1]÷2):(S[1]÷2), Ry in -(S[2]÷2):(S[2]÷2), Rz in -(S[3]÷2):(S[3]÷2)
                R = SVector{3,Int}(Rx,Ry,Rz)
                rj = frac_to_cart(lattice, fracpos[j] .+ SVector{3,Float64}(R))
                rij = rj - ri
                d2 = dot(rij, rij)
                (d2 == 0.0 || d2 > cutoff2) && continue
                kL_ = kL isa Function ? kL(i,j,rij) : float(kL)
                kT_ = kT isa Function ? kT(i,j,rij) : float(kT)
                push!(bonds, Bond(i, j, R, rij, kL_, kT_))
            end
        end
    end

    # Deduplicate (canonical ordering)
    keep = trues(length(bonds))
    for (idx,b) in pairs(bonds)
        if b.i == b.j && b.R == SVector{3,Int}(0,0,0)
            keep[idx] = false; continue
        end
        if b.i > b.j || (b.i == b.j && any(b.R .< 0))
            keep[idx] = false
        end
    end
    return [bonds[i] for i in eachindex(bonds) if keep[i]]
end

"""
    neighbor_bonds_from_sunny(cryst, bonds; kL=1.0, kT=1.0)

Fast-path bond construction using Sunny's bonds and geometry.
`bonds` may be a `Vector` of:
- Sunny.Bond         (fields: i,j,n)
- Tuples (i,j,n::SVector{3,Int})
- Tuples (i,j,Δ::SVector{3,Float64})  # direct Cartesian Δ

`kL`/`kT` can be numbers or `(i,j,r0)->value`. Returns `Vector{Bond}`.
"""
function neighbor_bonds_from_sunny(cryst, bonds;
                                   kL::Union{Real,Function}=1.0,
                                   kT::Union{Real,Function}=1.0)
    L = SMatrix{3,3,Float64,9}(Matrix(getproperty(cryst, :latvecs)))
    out = Bond{Float64}[]
    S = isdefined(Main, :Sunny) ? _sunny_mod() : nothing

    for nb in bonds
        if S !== nothing && (hasproperty(nb, :i) && hasproperty(nb, :j) && hasproperty(nb, :n))
            i = getproperty(nb, :i); j = getproperty(nb, :j); n = getproperty(nb, :n)
            # Use Sunny's precise geometry
            bpos = S.BondPos(cryst, nb)
            r0   = SVector{3,Float64}(S.global_displacement(cryst, bpos))
            R    = SVector{3,Int}(n)
        elseif nb isa Tuple{Int,Int,SVector{3,Int}}
            i,j,R = nb
            # Need fractional positions for j + R - i; fetch from cryst if available
            fpos = getproperty(cryst, :positions)
            r0   = SVector{3,Float64}(L * (SVector{3,Float64}(fpos[j]) .+ SVector{3,Float64}(R) .- SVector{3,Float64}(fpos[i])))
        elseif nb isa Tuple{Int,Int,SVector{3,Float64}}
            i,j,Δ = nb
            R     = SVector{3,Int}(0,0,0)
            r0    = Δ
        else
            throw(ArgumentError("Unsupported neighbor element of type $(typeof(nb))"))
        end

        d2 = dot(r0,r0)
        (d2 == 0.0) && continue
        kL_ = kL isa Function ? kL(i,j,r0) : float(kL)
        kT_ = kT isa Function ? kT(i,j,r0) : float(kT)
        push!(out, Bond(i,j,R,r0,kL_,kT_))
    end

    # Deduplicate (canonical ordering)
    keep = trues(length(out))
    for (idx,b) in pairs(out)
        if b.i == b.j && b.R == SVector{3,Int}(0,0,0)
            keep[idx] = false; continue
        end
        if b.i > b.j || (b.i == b.j && any(b.R .< 0))
            keep[idx] = false
        end
    end
    return [out[i] for i in eachindex(out) if keep[i]]
end

"""
    neighbor_bonds_radius(cryst; cutoff, kL, kT)

Sunny-powered radius query: builds bonds using `Sunny.all_bonds_for_atom`.
Requires `using Sunny`. Returns `Vector{Bond}`.
"""
function neighbor_bonds_radius(cryst; cutoff::Real, kL::Union{Real,Function}=1.0, kT::Union{Real,Function}=1.0)
    S = _sunny_mod()
    N = length(getproperty(cryst, :positions))
    bonds_sunny = S.Bond[]
    for i in 1:N
        append!(bonds_sunny, S.all_bonds_for_atom(cryst, i, cutoff))
    end
    return neighbor_bonds_from_sunny(cryst, bonds_sunny; kL=kL, kT=kT)
end

# ---------------------------
# Force-constant assembly
# ---------------------------

"""
    assemble_force_constants!(model)

Builds real-space force constants Φ as a dictionary of 3×3 blocks keyed by (i,j,R).
Per-bond block: `K = kL*(êêᵀ) + kT*(I - êêᵀ)` with `ê = r0/‖r0‖`, applied to (u_j@R - u_i@0).
Conservation laws are enforced later by `enforce_asr!`.
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

    return Φ
end

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

# ---------------------------
# Dynamical matrix & phonons (energies in meV)
# ---------------------------

const ℏ = 1.054571817e-34
const kB = 1.380649e-23

# Unit conversions
const eV_to_J      = 1.602176634e-19
const meV_to_J     = 1.602176634e-22
const u_to_kg      = 1.66053906660e-27
const N_per_m_per_eVA2 = 16.02176634              # 1 eV/Å^2 = 16.02176634 N/m
# If Φ built with k in eV/Å^2 and masses in u:
#   ω_int = sqrt( eV/Å^2 / u ),  E_meV = α * ω_int  with  α = ħ sqrt(N_per_m_per_eVA2/u_to_kg) / meV_to_J
const ALPHA_meV    = 64.65415130134122

# --- meV–Å–amu convenience constants ---
const HBAR_meV_ps        = 0.6582119514       # meV·ps
const K_B_meV_per_K      = 0.08617333262145   # meV/K
const AMU_Aps2_to_meV    = 0.103642691        # 1 amu·(Å/ps)^2 = 0.103642691 meV
const MSD_PREF_A2        = (HBAR_meV_ps^2) / (2*AMU_Aps2_to_meV)  # Å^2

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

"""
    phonons(model, Φ, q; q_basis=:cart, q_cell=:primitive)

Diagonalize D(q). Returns (E, Evec) with **energies in meV** and 3N×3N eigenvectors.
`q` may be in Cartesian Å⁻¹ (`q_basis=:cart`) or RLU (`q_basis=:rlu` with `q_cell=:primitive` or `:conventional`).
If passing RLU, also pass the underlying crystal `cryst` via keyword.
"""
function phonons(model::Model, Φ, q::SVector{3,Float64}; q_basis::Symbol=:cart, q_cell::Symbol=:primitive, cryst=nothing)
    q_cart = q_basis === :cart ? q :
             (cryst === nothing ? error("cryst required for q_basis=:rlu") :
              q_cartesian(cryst, q; basis=:rlu, cell=q_cell))
    Dq = dynamical_matrix(model, Φ, q_cart)
    vals, vecs = eigen(Dq)
    ω2 = max.(real(vals), 0.0)
    ω  = sqrt.(ω2)                    # internal sqrt(k/m) units
    E  = ALPHA_meV .* ω               # energies in meV (ℏω)
    perm = sortperm(E)
    return E[perm], vecs[:,perm]
end


# ---------------------------
# Mean-square displacements & isotropic Debye–Waller from phonons
# ---------------------------

"""
    msd_from_phonons(model, Φ; T, cryst, qgrid=(12,12,12), q_cell=:primitive, eps_meV=1e-6)

Compute per-site mean-square displacements ⟨u_s^2⟩(T) in Å² directly from the phonon spectrum:
⟨u_s^2⟩ = (1/Nq) * Σ_{q,ν} (ħ² / (2 M_s E_J)) * coth(E_J / (2 k_B T)) * Σ_α |e_phys(s,α;q,ν)|²
where E_J = E_meV ⋅ meV_to_J and e_phys are physical (non mass-weighted) polarization vectors.
Modes with E ≤ eps_meV are skipped (Γ handling); refine qgrid for convergence.
"""
function msd_from_phonons(model::Model, Φ;
                          T::Real, cryst, qgrid::NTuple{3,Int}=(12,12,12),
                          q_cell::Symbol=:primitive, eps_meV::Real=1e-6)
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

"""
    B_isotropic_from_phonons(model, Φ; T, cryst, qgrid=(12,12,12), q_cell=:primitive, eps_meV=1e-6)

Return isotropic Debye–Waller B-factors (Å²) per site from first principles:
B_s(T) = 8π² ⟨u_s^2⟩(T).
"""
function B_isotropic_from_phonons(model::Model, Φ;
                                  T::Real, cryst, qgrid::NTuple{3,Int}=(12,12,12),
                                  q_cell::Symbol=:primitive, eps_meV::Real=1e-6)
    msd = msd_from_phonons(model, Φ; T=T, cryst=cryst, qgrid=qgrid, q_cell=q_cell, eps_meV=eps_meV)
    return (8π^2) .* msd
end

# Full anisotropic displacement tensors U^{(s)}_{αβ} from phonons (Å^2)

function U_from_phonons(model::Model, Φ;
                        T::Real, cryst, qgrid::NTuple{3,Int}=(12,12,12),
                        q_cell::Symbol=:primitive, eps_meV::Real=1e-6)
    N = model.N
    nx, ny, nz = qgrid
    Nq = nx*ny*nz
    U = [zeros(SMatrix{3,3,Float64,9}) for _ in 1:N]

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
                e1 = Evec[i1, ν]; e2 = Evec[i2, ν]; e3 = Evec[i3, ν]
                # physical polarization tensor components: (eα eβ*)/M_s
                M = model.mass[s]
                t11 = (e1*conj(e1))/M; t12 = (e1*conj(e2))/M; t13 = (e1*conj(e3))/M
                t21 = (e2*conj(e1))/M; t22 = (e2*conj(e2))/M; t23 = (e2*conj(e3))/M
                t31 = (e3*conj(e1))/M; t32 = (e3*conj(e2))/M; t33 = (e3*conj(e3))/M
                Tmat = @SMatrix [t11 t12 t13; t21 t22 t23; t31 t32 t33]
                U[s] += (MSD_PREF_A2 * cothx / EmeV) * real.(Tmat)
            end
        end
    end
    for s in 1:N
        U[s] = U[s] ./ Nq
    end
    return U
end

# ---------------------------
# One-phonon coherent DSF (energy in meV)
# ---------------------------

"""
    onephonon_dsf(model, Φ, q, Evals; T=300.0, bcoh=nothing, B=nothing, η=0.5, mass_unit=:amu, q_basis=:cart, q_cell=:primitive, cryst=nothing)

Computes one-phonon coherent S(q,E) on energy grid `Evals` (meV).
- Debye–Waller is intrinsic: full anisotropic tensors U_s(T) are computed from the phonon spectrum (no Θ_D).
- `bcoh` optional Vector of coherent scattering lengths (length N)
- `η` Gaussian HWHM for energy broadening (meV)
- `mass_unit` :amu or :kg (Model.mass is assumed to be :amu typically)
- `q_basis`/`q_cell`/`cryst` follow `phonons` conventions
Returns S(E) (arbitrary but consistent units).
"""
function onephonon_dsf(model::Model, Φ, q::SVector{3,Float64}, Evals::AbstractVector{<:Real};
                        T::Real=300.0, bcoh=nothing, η::Real=0.5, mass_unit::Symbol=:amu,
                        q_basis::Symbol=:cart, q_cell::Symbol=:primitive, cryst=nothing,
                        dw_qgrid::NTuple{3,Int}=(12,12,12), _U_internal::Union{Nothing,Vector{SMatrix{3,3,Float64,9}}}=nothing)
    N = model.N
    M = model.mass
    # Use masses in amu regardless of user storage
    Mamu = mass_unit === :amu ? M : mass_unit === :kg ? (M ./ u_to_kg) : error("mass_unit must be :amu or :kg")
    bw = bcoh === nothing ? ones(N) : bcoh
    qvec = q_basis === :cart ? q : q_cartesian(cryst, q; basis=:rlu, cell=q_cell)
    U = _U_internal === nothing ? U_from_phonons(model, Φ; T=T, cryst=cryst, qgrid=dw_qgrid, q_cell=q_cell) : _U_internal
    DW = similar(Mamu)
    for s in 1:N
        DW[s] = exp(-dot(qvec, U[s]*qvec))
    end

    # Energies in meV from phonons(); eigenvectors are mass-weighted
    Eν, Evec = phonons(model, Φ, q; q_basis=q_basis, q_cell=q_cell, cryst=cryst)

    # Bose factor in meV units
    nB = @. 1/(exp(Eν/(K_B_meV_per_K*T)) - 1)

    Sω = zeros(Float64, length(Evals))
    for ν in 1:length(Eν)
        EmeV = Eν[ν]
        EmeV <= 0 && continue
        proj = 0.0
        for s in 1:N
            i1 = 3s - 2; i2 = 3s - 1; i3 = 3s
            # physical polarization
	    e1 = Evec[i1, ν]; e2 = Evec[i2, ν]; e3 = Evec[i3, ν]
            qdot = abs2(qvec[1]*e1 + qvec[2]*e2 + qvec[3]*e3)
            proj += (bw[s]^2 * DW[s] * qdot) #/ (2 * Mamu[s] * EmeV)
        end

        # Gaussian broadened delta in ENERGY domain (meV)
        @inbounds @simd for k in eachindex(Evals)
            E = Evals[k]
            Sω[k] += proj * (nB[ν] + 1) * exp(-((E - EmeV)^2)/(2η^2)) / (sqrt(2π)*η)
        end
    end
    return Sω
end


# ---------------------------
# Element/isotope data & lookups
# ---------------------------

# Natural-abundance atomic masses (amu)
const MASS_U = Dict{Symbol,Float64}(
    :H => 1.00794, :He => 4.002602, :C => 12.011, :N => 14.0067, :O => 15.999,
    :F => 18.998403163, :Ne => 20.1797, :Na => 22.98976928, :Mg => 24.305, :Al => 26.9815385,
    :Si => 28.085, :P => 30.973761998, :S => 32.06, :Cl => 35.45, :Ar => 39.948,
    :K => 39.0983, :Ca => 40.078, :Ti => 47.867, :V => 50.9415, :Cr => 51.9961,
    :Mn => 54.938044, :Fe => 55.845, :Co => 58.933194, :Ni => 58.6934, :Cu => 63.546,
    :Zn => 65.38, :Ga => 69.723, :Ge => 72.630, :As => 74.921595, :Se => 78.971,
)

# Isotopic masses (amu) for common isotopes used in tests/examples.
const MASS_ISO_U = Dict{Tuple{Symbol,Int},Float64}(
    (:H,1)=>1.00782503223, (:H,2)=>2.01410177812, (:H,3)=>3.01604928199,
    (:C,12)=>12.0, (:C,13)=>13.00335483507,
    (:N,14)=>14.00307400443, (:N,15)=>15.00010889888,
    (:O,16)=>15.99491461957, (:O,17)=>16.99913175650, (:O,18)=>17.99915961286,
    (:Si,28)=>27.97692653465, (:Si,29)=>28.97649466490, (:Si,30)=>29.973770136,
    (:Cu,63)=>62.929597474, (:Cu,65)=>64.92778970,
    (:Fe,54)=>53.93960899, (:Fe,56)=>55.93493633, (:Fe,57)=>56.93539284, (:Fe,58)=>57.93327443,
)

# Neutron coherent scattering lengths (fm): natural abundance (approx)
const BCOH_FM = Dict{Symbol,Float64}(
    :H => -3.7390, :D => 6.671, :C => 6.6460, :N => 9.36, :O => 5.803,
    :Si => 4.1491, :Fe => 9.45, :Cu => 7.718, :Ni => 10.3, :O16 => 5.803, :O18 => 5.841,
)

# Isotopic coherent lengths (fm) (limited subset for examples)
const BCOH_ISO_FM = Dict{Tuple{Symbol,Int},Float64}(
    (:H,1)=> -3.7390, (:H,2)=>6.671,  # H, D
    (:C,12)=>6.6511, (:C,13)=>6.19,
    (:O,16)=>5.803,  (:O,18)=>5.841,
    (:Si,28)=>4.106, (:Si,29)=>4.70, (:Si,30)=>6.50,
)

# Parse species labels, possibly with isotope mass number (e.g., "Si-29", "O18"), also "D","T"
function _parse_species_label(x)::Tuple{Symbol,Union{Int,Nothing}}
    s = String(x); s = strip(s)
    if s == "D"; return (:H, 2) end
    if s == "T"; return (:H, 3) end
    m = match(r"^([A-Za-z]{1,2})(?:-|\s*)?(\d+)?$", s)
    if m === nothing
        return (Symbol(s), nothing)
    else
        Z = Symbol(m.captures[1])
        A = m.captures[2] === nothing ? nothing : parse(Int, m.captures[2])
        return (Z, A)
    end
end

function mass_lookup(species::AbstractVector; iso_by_site::Dict{Int,Int}=Dict{Int,Int}(), iso_by_species::Dict{Symbol,Int}=Dict{Symbol,Int}())
    n = length(species)
    out = Vector{Float64}(undef, n)
    for s in 1:n
        Zs, A_tag = _parse_species_label(species[s])
        A = haskey(iso_by_site, s) ? iso_by_site[s] :
            (A_tag !== nothing ? A_tag :
             (haskey(iso_by_species, Zs) ? iso_by_species[Zs] : nothing))
        if A !== nothing
            key = (Zs, A)
            haskey(MASS_ISO_U, key) || error("No isotopic mass for $(Zs)-$(A)")
            out[s] = MASS_ISO_U[key]
        else
            haskey(MASS_U, Zs) || error("No natural-abundance mass for element $(Zs)")
            out[s] = MASS_U[Zs]
        end
    end
    return out
end

function bcoh_lookup(species::AbstractVector; iso_by_site::Dict{Int,Int}=Dict{Int,Int}(), iso_by_species::Dict{Symbol,Int}=Dict{Symbol,Int}())
    n = length(species)
    out = Vector{Float64}(undef, n)
    for s in 1:n
        Zs, A_tag = _parse_species_label(species[s])
        A = haskey(iso_by_site, s) ? iso_by_site[s] :
            (A_tag !== nothing ? A_tag :
             (haskey(iso_by_species, Zs) ? iso_by_species[Zs] : nothing))
        if A !== nothing
            key = (Zs, A)
            if haskey(BCOH_ISO_FM, key)
                out[s] = BCOH_ISO_FM[key]
            else
                haskey(BCOH_FM, Zs) || error("No coherent length for element $(Zs)")
                out[s] = BCOH_FM[Zs]
            end
        else
            if haskey(BCOH_FM, Zs)
                out[s] = BCOH_FM[Zs]
            else
                # Allow symbols like :O16 in the table
                if haskey(BCOH_FM, Symbol(string(Zs)))
                    out[s] = BCOH_FM[Symbol(string(Zs))]
                else
                    error("No coherent length for element $(Zs)")
                end
            end
        end
    end
    return out
end
# ---------------------------
# Model construction helpers
# ---------------------------

# Extract Sunny-like information without taking a hard dependency on Sunny.jl.
@inline function _to_phunny_spec(crystal)
    if hasproperty(crystal, :latvecs) && hasproperty(crystal, :positions) && hasproperty(crystal, :types)
        L  = SMatrix{3,3,Float64,9}(Matrix(getproperty(crystal, :latvecs)))
        fp = [SVector{3,Float64}(Tuple(p)) for p in getproperty(crystal, :positions)]
        sp = Symbol.(getproperty(crystal, :types))
        return (lattice=L, positions=fp, species=sp)
    else
        L  = SMatrix{3,3,Float64,9}(getproperty(crystal, :lattice))
        fp = [SVector{3,Float64}(p) for p in getproperty(crystal, :positions)]
        sp = Symbol.(getproperty(crystal, :species))
        return (lattice=L, positions=fp, species=sp)
    end
end

"""
    build_model(crystal; mass, neighbors_sunny=nothing, neighbors=nothing, cutoff=nothing, use_sunny_radius=true, kL=1.0, kT=1.0, supercell=(1,1,1))

Create a `Model` from a Sunny-like `crystal` object. Minimal interface expected:
- Either Sunny-style fields: `crystal.latvecs` (3×3), `crystal.positions` (fractional), `crystal.types` (Strings)
- or generic fields: `crystal.lattice` (3×3), `crystal.positions` (fractional), `crystal.species` (Symbols/Strings)

Bond assembly precedence:
1. If `neighbors_sunny` (e.g. `Vector{Sunny.Bond}`) is provided → fast path via Sunny geometry.
2. Else if `neighbors` (list of tuples as documented in `neighbor_bonds_from_sunny`) is provided → fast tuple path.
3. Else if `use_sunny_radius && cutoff!=nothing && Sunny loaded` → build via `Sunny.all_bonds_for_atom`.
4. Else → fallback to portable cutoff scan `neighbor_bonds_cutoff`.
"""
function build_model(crystal; mass=:lookup, isotopes_by_site=nothing, isotopes_by_species=nothing,
                     neighbors_sunny=nothing, neighbors=nothing, cutoff=nothing, use_sunny_radius::Bool=true, kL=1.0, kT=1.0, supercell=(1,1,1))
    spec = _to_phunny_spec(crystal)
    L, fpos, species = spec.lattice, spec.positions, spec.species

    massvec = if mass === :lookup
        mass_lookup(species;
            iso_by_site = isotopes_by_site === nothing ? Dict{Int,Int}() : isotopes_by_site,
            iso_by_species = isotopes_by_species === nothing ? Dict{Symbol,Int}() : isotopes_by_species)
    elseif mass isa Dict
        [haskey(mass, s) ? Float64(mass[s]) :
         haskey(mass, String(s)) ? Float64(mass[String(s)]) :
         error("No mass for species $s in provided mass mapping") for s in species]
    else
        collect(Float64.(mass))
    end

    bonds = nothing
    if neighbors_sunny !== nothing
        bonds = neighbor_bonds_from_sunny(crystal, neighbors_sunny; kL=kL, kT=kT)
    elseif neighbors !== nothing
        bonds = neighbor_bonds_from_sunny((; latvecs=Matrix(L), positions=fpos), neighbors; kL=kL, kT=kT)
    elseif use_sunny_radius && cutoff !== nothing && isdefined(Main, :Sunny)
        bonds = neighbor_bonds_radius(crystal; cutoff=cutoff, kL=kL, kT=kT)
    else
        cutoff === nothing && error("Provide `cutoff` if `bonds` not given (or pass `neighbors_sunny`/`neighbors`).")
        bonds = neighbor_bonds_cutoff(L, fpos; cutoff=cutoff, kL=kL, kT=kT, supercell=supercell)
    end

    return Model(L, fpos, species, massvec, bonds, length(fpos))
end

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

# ---------------------------
# 4D S(q, w) over a Cartesian or RLU grid
# ---------------------------
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

This routine is thread-parallel over the q-grid.
"""
function onephonon_dsf_4d(model::Model, Φ,
                          q1::AbstractVector, q2::AbstractVector, q3::AbstractVector,
                          Evals::AbstractVector;
                          q_basis::Symbol=:rlu, q_cell::Symbol=:primitive, cryst=nothing,
                          T::Real=300.0, η::Real=0.5, mass_unit::Symbol=:amu, bcoh=nothing,
                          threads::Bool=true, dw_qgrid::NTuple{3,Int}=(12,12,12))

    n1, n2, n3, nE = length(q1), length(q2), length(q3), length(Evals)
    S4 = zeros(Float64, n1, n2, n3, nE)

    # Precompute anisotropic Debye–Waller tensors once (Å^2)
    U = U_from_phonons(model, Φ; T=T, cryst=cryst, qgrid=dw_qgrid, q_cell=q_cell)

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
        Sq = onephonon_dsf(model, Φ, qcart, Evals; T=T, bcoh=bcoh,
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
#------------------
# S(q,w) Reshaping
#------------------
const AX = (; h=1, k=2, ℓ=3, l=3, ω=4, w=4)
function collapse(A; over=:ω, op=sum)
	axes = over isa Tuple ? over : (over,)
	idxs = sort!(map(x -> x isa Symbol ? AX[x] : Int(x), collect(axes)))
	for ax in Iterators.reverse(idxs)       # reduce highest axis first
		A = dropdims(op(A; dims=ax); dims=ax)
	end
	return A
end
end # module Phunny
