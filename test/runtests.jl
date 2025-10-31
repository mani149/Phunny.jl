using LinearAlgebra, StaticArrays, SparseArrays, Phunny, Sunny
using Test

# Silicon (diamond structure)
a = 5.431
lat = lattice_vectors(a, a, a, 90, 90, 90) # Sunny func
pos = [[0,0,0], [1/4,1/4,1/4]]
types = ["Si","Si"]
cryst = Crystal(lat, pos; types) # Sunny func

# Lookup masses and coherent scattering lengths
mass  = mass_lookup(Symbol.(types)) 	     # Dict(String=>amu)
bcoh  = bcoh_lookup(Symbol.(types))           # Vector per site (fm)

# Springs (toy Si-like constants)
kL = (i,j,rij)->175.0      # eV/Å^2
kT = (i,j,rij)->127.0

mdl = build_model(cryst; cutoff=2*a, use_sunny_radius=false,
                         kL=kL, kT=kT)
Φ   = assemble_force_constants!(mdl)
enforce_asr!(Φ, mdl.N)

# 4D DSF grid
h = collect(range(-5.5, 5.5; length=101))
k = collect(range(-5.5, 5.5; length=101))
l = [0.0]
E = ω_grid(0.0, 200.0, 401)
T = 150.0
η = 3.0

S4 = onephonon_dsf_4d(mdl, Φ, h, k, l, E;
                             q_basis=:rlu, q_cell=:primitive,
                             cryst=cryst, T=T, η=η,
                             mass_unit=:amu, bcoh=bcoh, threads=false)

# Unit tests
@testset "onephonon" begin
	println()
	@test size(S4) == (101, 101, 1, 401) 	# 4d
	@test all(isfinite, S4) 		# 4d
end

@testset "grid" begin
	println()
	@test E == 0.0:0.5:200.0
end

@testset "build_model" begin
	println()
	@test mdl.bonds[1].i == 1
	@test mdl.bonds[1].j == 2
	@test mdl.bonds[1].r0 ≈ [1.35775, 1.35775, 1.35775]
	@test mdl.bonds[1].R ≈ [0, 0, 0]
    @test mdl.lattice ≈ [5.431  0.0    0.0;
			 0.0    5.431  0.0;
 			 0.0    0.0    5.431]
    @test mdl.species == [:Si, :Si]
    @test mdl.N == 2
end;

@testset "force_constants" begin
	println()
	@test Φ.idxfloor == 1
	@test Φ.count == 4
	@test Φ.vals.length == 16
	@test size(Φ.vals[1]) == (3,3)
end

@testset "mass_bcoh" begin
	println()
	@test mass ≈ [28.085, 28.085]
	@test bcoh ≈ [4.1491, 4.1491]
end

eigvals, eigvecs = phonons(mdl, Φ, @SVector[0.5,0.5,0.5]; q_basis=:cart, q_cell=:primitive)
