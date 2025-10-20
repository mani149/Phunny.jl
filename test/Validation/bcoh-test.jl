using .Phunny, StaticArrays, Test, LinearAlgebra

# --- Minimal Model Constructor for bcoh Checks Only --- #
function _toy_model(species::Vector{Symbol})
	N = length(species)
	lattice = SMatrix{3,3,Float64,9}(I) #Dummy lattice (not used)
	fracpos = [@SVector[0.0, 0.0, 0.0] for _ in 1:N]
	mass = fill(28.085, N) #Dummy masses (not used)
	bonds = Phunny.Bond{Float64}[] #Empty Bond List
	return Phunny.Model(lattice, fracpos, species, mass, bonds, N)
end

# --- Unit Tests for Automatic Coherent-Scattering Length Resolution --- #
@testset "bcoh auto-lookup resolution" begin
	#Choose an element with entries in both BCOH_FM and BCOH_ISO_FM
	species = [:Si, :Si]
	m = _toy_model(species)

	#1) Auto-lookup (natural abundance)
	bw_auto = Phunny._resolve_bcoh(m; bcoh=nothing)
	@test length(bw_auto) == 2
	@test all(isfinite, bw_auto)
	#Reference from constants: BCOH_FM[:Si]
	@test isapprox(bw_auto, fill(Phunny.BCOH_FM[:Si], 2); rtol=0, atol=0)
	
	#2) Scalar/Vector Tests
	bw_scalar = Phunny._resolve_bcoh(m; bcoh=3.14)
	@test bw_scalar == [3.14, 3.14]
	bw_vec_input = [1.0, 2.0]
	bw_vec = Phunny._resolve_bcoh(m; bcoh=bw_vec_input)
	@test bw_vec == bw_vec_input

	#3) Isotope Override Tests
	#	(a) by species -> both sites set to same isotope
	bw_iso_species = Phunny._resolve_bcoh(m; bcoh=nothing, iso_by_species=Dict(:Si=>28))
	@test bw_iso_species == fill(Phunny.BCOH_ISO_FM[(:Si,28)],2)
	#	(b) by site -> site override wins over species override for particular index
	bw_iso_both = Phunny._resolve_bcoh(m; bcoh=nothing, 
					   iso_by_species=Dict(:Si=>30), 
					   iso_by_site=Dict(2=>29))
	@test bw_iso_both[1] == Phunny.BCOH_ISO_FM[(:Si,30)]
	@test bw_iso_both[2] == Phunny.BCOH_ISO_FM[(:Si,29)]

	#4) Length mismatch error for explicitly defined vector
	@test_throws ErrorException Phunny._resolve_bcoh(m; bcoh=[1.0]) #wrong length
end
print("bcoh auto-lookup tests: OK")
