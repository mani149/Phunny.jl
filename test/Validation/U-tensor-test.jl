using Test, LinearAlgebra, StaticArrays, Sunny, .Phunny

# --- Build Crystal & Model --- #
a = 5.43; L = lattice_vectors(a,a,a,90,90,90) #Conventional Cubic
fpos = [@SVector[0.0, 0.0, 0.0], @SVector[0.25, 0.25, 0.25]]; types = ["Si", "Si"] 
cryst = Crystal(L, fpos; types=types)

cutoff = a*(sqrt(3)/4) #Target NN Bonds
kL = 75.0; kT = 27.0
model = build_model(cryst; cutoff=cutoff, kL=kL, kT=kT)
FCMs = assemble_force_constants!(model)
enforce_asr!(FCMs, model.N)

# --- Useful Constants --- #
const T1, T2 = 300.0, 600.0
const grid1, grid2 = (8,8,8), (12,12,12)

# --- Quick Sanity Check for Mass --- #
@assert minimum(model.mass) > 0 && maximum(model.mass) > 1e-5 "model.mass must be in amu; got $(extrema(model.mass))"

# --- Anisotropic U Tests --- #
@testset "U: Real Symmetric & Positive Semidefinite" begin
	U = U_from_phonons(model, FCMs; T=T1, cryst=cryst, qgrid=grid1, q_cell=:conventional)
	@test length(U) == model.N
	for s in 1:model.N
		Us = Matrix(U[s])
		@test isapprox(Us, Us', rtol=0, atol=1e-12)
		λ = eigvals(Symmetric(Us))
		@test minimum(λ) ≥ -1e-12
	end
end

@testset "U: Cubic-site Isotropy (Si)" begin
	U = U_from_phonons(model, FCMs; T=T1, cryst=cryst, qgrid=grid2, q_cell=:conventional)
	for s in 1:model.N
		Us = Matrix(U[s])
		off = abs(Us[1,2]) + abs(Us[1,3]) + abs(Us[2,3])
		@test off ≤ 5e-4
		@test isapprox(Us[1,1], Us[2,2]; rtol=1e-2)
		@test isapprox(Us[2,2], Us[3,3]; rtol=1e-2)
	end
end

@testset "U: (T -> 0) Zero-Point Motion" begin
	U1 = U_from_phonons(model, FCMs; T=1e-3, cryst=cryst, qgrid=grid1, q_cell=:conventional)
	U2 = U_from_phonons(model, FCMs; T=0.1,  cryst=cryst, qgrid=grid1, q_cell=:conventional)
	for s in 1:model.N
		@test isapprox(tr(Matrix(U1[s])), tr(Matrix(U2[s])); rtol=1e-3)
	end
end

@testset "U: (T -> Inf) High-T Scaling" begin
	U1 = U_from_phonons(model, FCMs; T=T1*T1, cryst=cryst, qgrid=grid1, q_cell=:conventional)
	U2 = U_from_phonons(model, FCMs; T=T2*T2, cryst=cryst, qgrid=grid1, q_cell=:conventional)
	r = sum(tr.(Matrix.(U2))) / sum(tr.(Matrix.(U1)))
	@test isapprox(r, (T2*T2)/(T1*T1); rtol=5e-2)
end

@testset "U: Brillouin Zone Grid Convergence" begin
	U1 = U_from_phonons(model, FCMs; T=T1, cryst=cryst, qgrid=grid1, q_cell=:conventional)
	U2 = U_from_phonons(model, FCMs; T=T1, cryst=cryst, qgrid=grid2, q_cell=:conventional)
	frob = 0.0 #Global Frobenius Norm = 0 : BZ Sampling Numerically Convergent
	for s in 1:model.N
		Δ = Matrix(U1[s]) - Matrix(U2[s])
		frob += sum(abs2,Δ)
	end
	@test sqrt(frob) ≤ 1e-3
end

@testset "U: Tr(U) = MSD" begin
	U = U_from_phonons(model, FCMs; T=T1, cryst=cryst, qgrid=grid1, q_cell=:conventional)
	msd = msd_from_phonons(model, FCMs; T=T1, cryst=cryst, qgrid=grid1, q_cell=:conventional)
	for s in 1:model.N
		@test isapprox(msd[s], tr(Matrix(U[s])); rtol=5e-3, atol=5e-5)
	end
end
print("Anisotropic Displacement Tensor: Tests Passed!")
