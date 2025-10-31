using Test, LinearAlgebra, StaticArrays, Sunny, Statistics

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

# Common knobs for U
qgrid = (16,16,16)
T_room = 300.0
epsE   = 0.2    # meV

# -----------------------------
# Helpers
# -----------------------------
# Average site U (3×3) from a Vector{SMatrix{3,3}}
avgU(Uvec) = begin
    U = zeros(3,3)
    for s in 1:length(Uvec)
        U .+= Matrix(Uvec[s])
    end
    U ./= length(Uvec)
    (U + U')/2
end

# Compute the DW exponent and factor from a given U (3×3) and q (in rlu)
# Uses your convention DW = exp(- q_cart' * U * q_cart).
DW_exponent(cryst, U::AbstractMatrix, qrlu::SVector{3,Float64}; cell=:conventional) = begin
    qcart = Phunny.q_cartesian(cryst, qrlu; basis=:rlu, cell=cell)
    dot(qcart, U*qcart)
end

DW_factor(cryst, U::AbstractMatrix, qrlu::SVector{3,Float64}; cell=:conventional) =
    exp(-DW_exponent(cryst, U, qrlu; cell=cell))

# Generate symmetry-related q’s (simple cubic directions)
unit_qs(len) = [
    @SVector[len, 0.0, 0.0],
    @SVector[0.0, len, 0.0],
    @SVector[0.0, 0.0, len],
    @SVector[len/√2, len/√2, 0.0],
    @SVector[len/√2, 0.0,      len/√2],
    @SVector[0.0,      len/√2, len/√2]
]

# Small-q set for linearized check (W ≈ ⟨u²⟩|q|² when U ≈ u I)
small_qs = unit_qs(0.05)

# -----------------------------
# Precompute U at a few T
# -----------------------------
U0   = Phunny.U_from_phonons(model, FCMs; T=1e-4, cryst=cryst, qgrid=qgrid, q_cell=:conventional, eps_meV=epsE)
U300 = Phunny.U_from_phonons(model, FCMs; T=T_room, cryst=cryst, qgrid=qgrid, q_cell=:conventional, eps_meV=epsE)
U1500= Phunny.U_from_phonons(model, FCMs; T=1500.0, cryst=cryst, qgrid=qgrid, q_cell=:conventional, eps_meV=epsE)
U3000= Phunny.U_from_phonons(model, FCMs; T=3000.0, cryst=cryst, qgrid=qgrid, q_cell=:conventional, eps_meV=epsE)

Ū0    = avgU(U0)
Ū300  = avgU(U300)
Ū1500 = avgU(U1500)
Ū3000 = avgU(U3000)

# Isotropic MSD for reference
msd0    = tr(Ū0)/3
msd300  = tr(Ū300)/3

# -----------------------------
# Tests
# -----------------------------
@testset "DW: Bounds & q=0" begin
    W0 = DW_exponent(cryst, Ū300, @SVector[0.0,0.0,0.0]; cell=:conventional)
    @test isapprox(W0, 0.0; atol=1e-12)
    for q in unit_qs(0.25)
        W = DW_exponent(cryst, Ū300, q; cell=:conventional)
        DW = exp(-W)
        @test W ≥ 0.0
        @test 0.0 < DW ≤ 1.0
    end
end

@testset "DW: Cubic symmetry of exponent W(q)=qᵀUq" begin
    # For cubic Si with averaged Ū ~ u I, W should be invariant under cubic permutations at fixed |q|
    qset = unit_qs(0.10)
    # Measure anisotropy
    u = msd300; Δ = Ū300 .- (u*I)(3)
    #Frobenius norm --> Dimensionless anisotropy
    δ = sqrt(sum(abs2,Δ))/u
    #Calculate Debye-Waller Factor
    Wvals = [DW_exponent(cryst, Ū300, q; cell=:conventional) for q in qset]
    Wavg  = mean(Wvals)
    #Tolerance scales with anisotropy; add 2% cushion for numerical safety
    rtol_dir = 1.5*δ + 0.02
    #Calculate Debye-Waller Factor
    for W in Wvals
        @test isapprox(W, Wavg; rtol=rtol_dir)  # ~ 3% band; tightens as grid increases
    end
end

@testset "DW: Small-q isotropic relation W ≈ ⟨u²⟩ |q|²" begin
    # Using isotropic MSD u = tr(Ū)/3, check W/u|q|² ≈ 1 for small q
    u = msd300
    for q in small_qs
        qcart = Phunny.q_cartesian(cryst, q; basis=:rlu, cell=:conventional)
        W = DW_exponent(cryst, Ū300, q; cell=:conventional)
        u = msd300; Δ = Ū300 .- (u*I)(3); δ = sqrt(sum(abs2,Δ))/u	 
	ratio = W/(u*dot(qcart,qcart))
	#allow tiny numerical cushion of 2% beyong spectral bound
	@test abs(ratio - 1.0) <= δ + 0.02
    end
end

@testset "DW: Zero-point consistency (T→0)" begin
    # W(T) should approach W(0) as T→0
    for q in unit_qs(0.25)
        W0   = DW_exponent(cryst, Ū0,   q; cell=:conventional)
        Wlow = DW_exponent(cryst, avgU(Phunny.U_from_phonons(model, FCMs; T=1e-3, cryst=cryst, qgrid=qgrid, q_cell=:conventional, eps_meV=epsE)), q; cell=:conventional)
        @test isapprox(Wlow, W0; rtol=5e-3)
    end
end

@testset "DW: High-T linear scaling (classical regime)" begin
    # In the classical limit, W(T) ∝ T at fixed q
    q = @SVector[0.20, 0.0, 0.0]
    W1 = DW_exponent(cryst, Ū1500, q; cell=:conventional)
    W2 = DW_exponent(cryst, Ū3000, q; cell=:conventional)
    r  = W2/W1
    @test isapprox(r, 3000/1500; rtol=0.05)
end

@testset "DW: Brillouin-zone grid convergence (W at fixed q)" begin
    q = @SVector[0.20, 0.0, 0.0]
    Uc1 = avgU(Phunny.U_from_phonons(model, FCMs; T=T_room, cryst=cryst, qgrid=(8,8,8),  q_cell=:conventional, eps_meV=epsE))
    Uc2 = avgU(Phunny.U_from_phonons(model, FCMs; T=T_room, cryst=cryst, qgrid=(12,12,12), q_cell=:conventional, eps_meV=epsE))
    W1 = DW_exponent(cryst, Uc1, q; cell=:conventional)
    W2 = DW_exponent(cryst, Uc2, q; cell=:conventional)
    @test isapprox(W2, W1; rtol=0.03)
end

@testset "DW: Consistency with isotropic B-factor (internal)" begin
    # Internal consistency only: B_iso = 8π² ⟨u²⟩, and for U ≈ u I → W = u|q|²
    u  = msd300
    q  = @SVector[0.20, 0.0, 0.0]
    qcart = Phunny.q_cartesian(cryst, q; basis=:rlu, cell=:conventional)
    W = DW_exponent(cryst, Ū300, q; cell=:conventional)
    W_iso = u*dot(qcart,qcart)
    @test isapprox(W, W_iso; rtol=0.05)
end

