# Tutorials

## Building / Analyzing a Phonon Model
First, define the lattice parameters and lattice geometry.
```julia
import Phunny, Sunny
a,b,c = 5.35/sqrt(2), 5.41/sqrt(2), 13.2 
L = Sunny.lattice_vectors(a, b, c, 90, 90, 90)
```
This example uses the primitive cell geometry for La₂CuO₄. There are seven atoms
for which we define a fractional position in the primitive cell and a corresponding element type.
```julia
#Small in-plane tilt applied to planar oxygens
δ = 0.0
#Fractional Positions + Atomic Labels
fpos = [@SVector[0.0,        0.0,      0.0  ], # Cu
        @SVector[0.5 + δ,    0.0,      0.0  ], # O ~ planar (x)
        @SVector[0.0,      0.5 + δ,    0.0  ], # O ~ planar (y)
        @SVector[0.0,        0.0,      0.183], # O ~ apical (+z)
        @SVector[0.0,        0.0,      0.817], # O ~ apical (-z)
        @SVector[0.0,        0.0,      0.361], # La (+z),
        @SVector[0.0,        0.0,      0.639]] # La (-z)
types = ["Cu", "O", "O", "O", "O", "La", "La"]
```
From here, populate the lattice with the atomic ions.
```julia
#Crystal + Model
cryst = Crystal(L, fpos; types)
```
To build the phonon model, a radial cut-off value must be defined for per-atom bonds within the primitive cell.
```julia
cutoff = 0.75*minimum((a,b,c)); 
model = build_model(cryst; cutoff)
```
For now, atomic force constants are manually defined with per-bond pairs.
```julia
#Manually Defined Bonds & Force Constants 
#      (:Atom₁, :Atom₂) => (kL, kT)
bond_dict = Dict( (:Cu, :O) => (4, 12),
                  (:O,  :O) => (3, 3),
                  (:O,  :La) => (0.2, 0.2))
```
Note, the bonds are reference with `:symbols`. Each atomic bond `(:atom1, :atom2)` is associated with a
longitudinal `kL` and transverse `kT` component with units meV/Å². Also, masses are implicitly defined in
units of AMU, unless specified. From here, assign force constants per bond, create a sparse force constant
matrix `ϕ` and enforce physical conservation laws.
```julia
assign_force_constants!(model, atoms, bond_dict)
ϕ = assemble_force_constants!(model); enforce_asr!(ϕ, model.N)
```
Now, you may solve the model for eigenfrequencies and phonon polarizations at a particular momentum-value.
```julia
eigenpairs = phonons(model, ϕ, @SVector[0.0, 0.0, 0.0]; q_basis=:rlu, q_cell=:primitive, cryst=cryst)
```
Additionally, you may calculate the dynamic structure factor along a particular q-space path.
```julia
#Choose your starting point and end point
q₀=@SVector[-1.0, 0.0, 0.0]; q₁=@SVector[1.0, 0.0, 0.0]

#Generate an array of nq points along the q-path
nq = 81; qs = [SVector{3,Float64}((1-t).*q₀ .+ t.*q₁) for t in range(0,1;length=nq)]

#Set an energy-transfer range
ωs = range(0.0, ωmax; length=1201)

#Calculate the one-phonon dynamic structure factor
Sqω = zeros(Float64, nq, nω)
@inbounds for (iq, q) in enumerate(qs)
      Sω = onephonon_dsf(model, Φ, q, ωs; q_basis=:rlu, q_cell=q_cell, cryst=cryst, T=0.25)
      Sqω[iq,:] .= Sω
end
```
## Using Sunny data

## Computing msd

