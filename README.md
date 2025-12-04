# Phunny

<!---
TODO: Add stable version 
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mani149.github.io/Phunny.jl/stable/)
 --->

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mani149.github.io/Phunny.jl/dev/)
[![Build Status](https://github.com/mani149/Phunny.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mani149/Phunny.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Semi-classical Phonon Calculations
Phunny is a Julia package for modeling phonons of a crystal lattice. It's written to integrate with the Sunny.jl API, a linear spin-wave theory package to model quantum magnetism. 
Phunny solves for the eigenfrequencies and phonon polarization vectors using a sparse representation of the force constant matrix, restricted to the unit/primitive cell. The force 
constant matrix can then be used to obtain the dynamic structure factor (DSF) and compared to experimental inelastic neutron scattering data. 
