using Test, LinearAlgebra, StaticArrays, Sunny

# Load your package
include(joinpath(@__DIR__, "../../src/Phunny.jl"))
using .Phunny

# -----------------------------
# Build crystal & force model
# -----------------------------
a = 5.43
L = lattice_vectors(a,a,a,90,90,90) # conventional cubic
fpos  = [@SVector[0.0,0.0,0.0], @SVector[0.25,0.25,0.25]]
types = ["Si","Si"]
cryst = Crystal(L, fpos; types=types)

cutoff = a*sqrt(3)/4
kL, kT = 75.0, 27.0
model = Phunny.build_model(cryst; cutoff=cutoff, kL=kL, kT=kT)
FCMs  = Phunny.assemble_force_constants!(model)
Phunny.enforce_asr!(FCMs, model.N)

# -----------------------------
# Helpers
# -----------------------------
# Trapezoidal integration on a uniform energy grid
trapz_uniform(y::AbstractVector{<:Real}, ΔE::Real) = ΔE * (sum(y) - 0.5*(first(y)+last(y)))

# Site-averaged U (3×3) from Vector{SMatrix{3,3}}
avgU(Uvec) = begin
    U = zeros(3,3)
    for s in 1:length(Uvec)
        U .+= Matrix(Uvec[s])
    end
    (U ./ length(Uvec) + (U ./ length(Uvec))')/2
end

# DW exponent/factor for a given 3×3 U and q (rlu)
DW_exponent(cryst, U::AbstractMatrix, qrlu::SVector{3,Float64}; cell=:conventional) = begin
    qcart = Phunny.q_cartesian(cryst, qrlu; basis=:rlu, cell=cell)
    dot(qcart, U*qcart)
end
DW_factor(cryst, U::AbstractMatrix, qrlu::SVector{3,Float64}; cell=:conventional) =
    exp(-DW_exponent(cryst, U, qrlu; cell=cell))

# Compute analytic area (zeroth moment) matching onephonon_dsf's integrand
function analytic_area_onephonon(model, Φ, q_rlu; T=300.0, cryst, q_cell=:conventional, Usite::Vector{SMatrix{3,3,Float64,9}})
    # masses in amu (match onephonon_dsf)
    Mamu = model.mass
    # bcoh resolution consistent with onephonon_dsf
    bw = Phunny._resolve_bcoh(model; bcoh=nothing, iso_by_site=nothing, iso_by_species=nothing)

    # q in Cartesian
    qvec = Phunny.q_cartesian(cryst, q_rlu; basis=:rlu, cell=q_cell)

    # Per-site DW factor at this q (same as onephonon_dsf)
    DWs = [exp(-dot(qvec, Matrix(Usite[s])*qvec)) for s in 1:model.N]

    # phonons at this q
    Eν, Evec = Phunny.phonons(model, Φ, q_rlu; q_basis=:rlu, q_cell=q_cell, cryst=cryst)
    nB = @. 1/(exp(Eν/(Phunny.K_B_meV_per_K*T)) - 1)

    A = 0.0
    for ν in eachindex(Eν)
        EmeV = Eν[ν]
        EmeV <= 0 && continue
        proj = 0.0
        @inbounds for s in 1:model.N
            i1 = 3s - 2; i2 = 3s - 1; i3 = 3s
            # mass-weighted eigenvector components (from phonons)
            e1 = Evec[i1, ν]; e2 = Evec[i2, ν]; e3 = Evec[i3, ν]
            qdot = abs2(qvec[1]*e1 + qvec[2]*e2 + qvec[3]*e3)
            proj += (bw[s]^2 * DWs[s] * qdot) / (2 * Mamu[s] * EmeV)
        end
        A += proj * (nB[ν] + 1)
    end
    return A
end

# -----------------------------
# Common knobs
# -----------------------------
qgrid_U = (16,16,16)          # good BZ sampling for U
epsE    = 0.2                 # meV
T_room  = 300.0
η1, η2  = 0.3, 1.0            # meV
Emax    = 120.0               # meV
ΔE      = 0.05                # meV
Egrid   = collect(-Emax:ΔE:Emax)

# Precompute U at 300 K for DW
Usite = Phunny.U_from_phonons(model, FCMs; T=T_room, cryst=cryst, qgrid=qgrid_U, q_cell=:conventional, eps_meV=epsE)
Ū     = avgU(Usite)

# Two small-q points along [100]
q1 = @SVector[0.10, 0.0, 0.0]   # rlu
q2 = @SVector[0.05, 0.0, 0.0]   # rlu

# -----------------------------
# Tests
# -----------------------------
@testset "S1: Area equals analytic sum over modes" begin
    q = q1
    Sη = Phunny.onephonon_dsf(model, FCMs, q, Egrid;
                               T=T_room, η=η1, mass_unit=:amu,
                               q_basis=:rlu, q_cell=:conventional, cryst=cryst,
                               dw_qgrid=qgrid_U, _U_internal=Usite)

    A_num = trapz_uniform(Sη, ΔE)
    A_th  = analytic_area_onephonon(model, FCMs, q; T=T_room, cryst=cryst, q_cell=:conventional, Usite=Usite)

    @test isapprox(A_num, A_th; rtol=0.03, atol=1e-12) 
    #If minimum(Egrid) >= 0, then A_num = 0.5*A_th (due to absence of anti-Stokes term) 
end

@testset "S2: Peak location matches phonon energy (narrow η)" begin
    q = q1
    Sη = Phunny.onephonon_dsf(model, FCMs, q, Egrid;
                               T=T_room, η=η1, mass_unit=:amu,
                               q_basis=:rlu, q_cell=:conventional, cryst=cryst,
                               dw_qgrid=qgrid_U, _U_internal=Usite)

    # Find strongest δ-like contribution analytically
    Eν, Evec = Phunny.phonons(model, FCMs, q; q_basis=:rlu, q_cell=:conventional, cryst=cryst)
    qcart = Phunny.q_cartesian(cryst, q; basis=:rlu, cell=:conventional)
    bw = Phunny._resolve_bcoh(model; bcoh=nothing, iso_by_site=nothing, iso_by_species=nothing)

    # Score each mode by its "area weight"
    weights = zeros(length(Eν))
    for ν in eachindex(Eν)
        EmeV = Eν[ν]; EmeV <= 0 && (weights[ν]=0; continue)
        nB = 1/(exp(EmeV/(Phunny.K_B_meV_per_K*T_room)) - 1)
        proj = 0.0
        for s in 1:model.N
            i1 = 3s-2; i2 = 3s-1; i3 = 3s
            e1 = Evec[i1,ν]; e2 = Evec[i2,ν]; e3 = Evec[i3,ν]
            qdot = abs2(qcart[1]*e1 + qcart[2]*e2 + qcart[3]*e3)
            proj += (bw[s]^2 * exp(-dot(qcart, Matrix(Usite[s])*qcart)) * qdot) / (2*model.mass[s]*EmeV)
        end
        weights[ν] = (nB+1)*proj
    end
    νmax = argmax(weights)
    Epeak_th = Eν[νmax]

    # Extract numeric peak location
    kmax = argmax(Sη)
    Epeak_num = Egrid[kmax]

    @test isapprox(Epeak_num, Epeak_th; atol=3η1)  # allow a few η of uncertainty on grid
end

@testset "S3: Area is independent of η" begin
    q = q1
    Sηa = Phunny.onephonon_dsf(model, FCMs, q, Egrid;
                                T=T_room, η=η1, mass_unit=:amu,
                                q_basis=:rlu, q_cell=:conventional, cryst=cryst,
                                dw_qgrid=qgrid_U, _U_internal=Usite)
    Sηb = Phunny.onephonon_dsf(model, FCMs, q, Egrid;
                                T=T_room, η=η2, mass_unit=:amu,
                                q_basis=:rlu, q_cell=:conventional, cryst=cryst,
                                dw_qgrid=qgrid_U, _U_internal=Usite)
    A1 = trapz_uniform(Sηa, ΔE)
    A2 = trapz_uniform(Sηb, ΔE)
    @test isapprox(A1, A2; rtol=0.02, atol=1e-12)
end

@testset "S4: Small-q scaling (acoustic-dominated)" begin
    # Expect A(q1)/A(q2) ≈ |q1|/|q2| = 2 within modest tolerance
    S1 = Phunny.onephonon_dsf(model, FCMs, q1, Egrid;
                               T=T_room, η=η1, mass_unit=:amu,
                               q_basis=:rlu, q_cell=:conventional, cryst=cryst,
                               dw_qgrid=qgrid_U, _U_internal=Usite)
    S2 = Phunny.onephonon_dsf(model, FCMs, q2, Egrid;
                               T=T_room, η=η1, mass_unit=:amu,
                               q_basis=:rlu, q_cell=:conventional, cryst=cryst,
                               dw_qgrid=qgrid_U, _U_internal=Usite)
    A1 = trapz_uniform(S1, ΔE)
    A2 = trapz_uniform(S2, ΔE)
    ratio = A1/A2
    @test isapprox(ratio, 2.0; rtol=0.25)
end

@testset "S5: Debye–Waller scaling of the area" begin
    q = q1
    # With computed U
    Swith = Phunny.onephonon_dsf(model, FCMs, q, Egrid;
                                  T=T_room, η=η1, mass_unit=:amu,
                                  q_basis=:rlu, q_cell=:conventional, cryst=cryst,
                                  dw_qgrid=qgrid_U, _U_internal=Usite)
    Awith = trapz_uniform(Swith, ΔE)

    # With DW artificially turned off by injecting zero U tensors
    Uzero = [zeros(SMatrix{3,3,Float64,9}) for _ in 1:model.N]
    SwoDW = Phunny.onephonon_dsf(model, FCMs, q, Egrid;
                                  T=T_room, η=η1, mass_unit=:amu,
                                  q_basis=:rlu, q_cell=:conventional, cryst=cryst,
                                  dw_qgrid=qgrid_U, _U_internal=Uzero)
    AwoDW = trapz_uniform(SwoDW, ΔE)

    # Expected scaling ≈ exp(-qᵀ Ū q) with site-average Ū (Si sites identical)
    DW̄ = DW_factor(cryst, Ū, q; cell=:conventional)
    @test isapprox(Awith/AwoDW, DW̄; rtol=0.05)
end
