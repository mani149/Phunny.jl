
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








