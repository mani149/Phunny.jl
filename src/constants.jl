"""
	Defining physical constants and data structures 
"""


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





















