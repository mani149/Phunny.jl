```@meta
CurrentModule = Phunny
```

# Phunny

Documentation for [Phunny](https://github.com/mani149/Phunny.jl).

## What's Phunny?

Phunny is a package for semi-classical phonon calculations, designed to integrate with the linear spin wave theory package Sunny. 


## Example Usage
```julia
using Phunny, Sunny, StaticArrays

#Build crystal with Sunny
a = 5.43; L = lattice_vectors(a,a,a,90,90,90)
fracpos = [@SVector[0.0, 0.0, 0.0],
           @SVector[0.25, 0.25, 0.25]]
types = ["Si", "Si"]
cryst = Crystal(L, fracpos; types=types)

#Build phonon model with Phunny
model = build_model(cryst; cutoff=0.45a, kL=10.0, kT=0.0)
FCMs = assemble_force_constants!(model); enforce_asr!(FCMs, model.N)

#Solve eigenvalue problem at Gamma point
q0 = @SVector[0.0, 0.0, 0.0]
eigvals, eigvecs = phonons(model, FCMs, q0; q_basis=:rlu) 

```

## Installation
Installation is currently possible via the Julia package registry.
```julia-repl
julia> ]
(v1.11) pkg> add https://github.com/mani149/Phunny.jl/

```
```julia-repl
julia> using Pkg; Pkg.add("https://github.com/mani149/Phunny.jl/")
```

