################################################################
# This script follows an example from M. P. Marder's           #
# Condensed Matter Physics textbook. The force constant        #
# is derived analytically for an ideal diatomic chain and      #
# compared with the results of Phunny.jl to determine accuracy.#
# 							       #
# 					      - Isaac C. Ownby #
################################################################
using LinearAlgebra, StaticArrays, SparseArrays, .Phunny, Sunny, Test, Plots

#---------------------------------------#
#   Build Model for 1D Diatomic Chain   #
#---------------------------------------#

a = 3.8  #Lattice Constant
d = 40.0

#Lattice Vectors (Cubic)
L = lattice_vectors(a, a, d, 90, 90, 90)

#Fractional Atomic Basis Positions
fpos = [@SVector[0.0, 0.0, 0.0], @SVector[0.5,0.0,0.0]]
types = ["Cu", "O"]

#Spring constants
kL, kT = 10.0, 10.0

#Build Phunny Model
cryst = Crystal(Matrix(L), fpos; types=types)
model = build_model(cryst; cutoff=0.6a, kL=kL, kT=kT)


#-----------------------------------#
#   Compute Force Constant Matrix   #
#-----------------------------------#

#Assemble FCMs
FCMs = assemble_force_constants!(model)

#Enforce acoustic sum rule
enforce_asr!(FCMs, model.N)

#-----------------------#
#   Analytic Solution   #
#-----------------------#

#Extract masses 
M = mass_vector(model) #units ~ atomic mass unit (amu)
m1, m2 = M[1], M[2]

#Converts frequency to meV
const alpha = Phunny.ALPHA_meV

#Analytic solution (J)
chain_dispersions(q) = begin
	s2 = sin(0.5*q*a)^2
	A = kL*( (1/m1) + (1/m2) )
	B = kL*sqrt( ((1/m1) + (1/m2))^2 - 4*s2/(m1*m2))
	(A - B, A + B)
end

#Analytic solution (meV)
chain_dispersions_meV(q) = begin
	Edn, Eup = chain_dispersions(q)
	(alpha*sqrt(max(Edn, 0.0)), alpha*sqrt(max(Eup, 0.0)))
end

#Compute eigenvalues/eigenvectors for x-polarized phonons (meV)
function xpolar_energies_meV(qx)
	E, V = phonons(model, FCMs, @SVector[qx, 0.0, 0.0]; q_basis=:cart)
	xwt = [sum(abs2(V[3*(s-1)+1, nu])/M[s] for s in 1:model.N) for nu in 1:length(E)]
	idx = sortperm(xwt;rev=true)[1:2]
	sort(E[idx])
end

#----------------------------------------------#
#   Compare Phunny.jl with analytic solution   #
#----------------------------------------------#
nh = 17
qs = range(-pi/a, stop=pi/a; length=nh)
num_LA = zeros(nh); num_LO = zeros(nh) #Numerical solution via Phunny.jl
ana_LA = zeros(nh); ana_LO = zeros(nh) #Analytic solution 

for (k,q) in enumerate(qs)
	num_LA[k], num_LO[k] = xpolar_energies_meV(q)
	ana_LA[k], ana_LO[k] = chain_dispersions_meV(q)
end

#---Error Analysis---#
abs_err_LA = maximum(abs.(num_LA .- ana_LA))
abs_err_LO = maximum(abs.(num_LO .- ana_LO))

rel_err_LA = maximum(k -> (ana_LA[k]>1e-8 ? 
			   abs((num_LA[k] - ana_LA[k])/ana_LA[k]) : 0.0), 1:nh)
rel_err_LO = maximum(k -> (ana_LO[k]>1e-8 ? 
			   abs((num_LO[k] - ana_LO[k])/ana_LO[k]) : 0.0), 1:nh)

#---Print Report---#
@info "Diatomic chain benchmark (Cu-O): kL = $(kL) eV/Å², a = $(a) Å"
print("\nMax Absolute Error: (LA) $(abs_err_LA) meV  |  (LO) $(abs_err_LO) meV \n")
print("\nMax Relative Error: (LA) $(rel_err_LA) meV  |  (LO) $(rel_err_LO) meV \n\n")

#----Gamma Point Selection Rule----#
b = bcoh_lookup(Symbol.(types)) #fm
acoustic_amplitude = b[1]/sqrt(m1) + b[2]/sqrt(m2)
optical_amplitude = b[1]/sqrt(m1) - b[2]/sqrt(m2)
print("Near Gamma Amplitude Ratio (Acoustic | Optical) : $(abs2(acoustic_amplitude)) | $(abs2(optical_amplitude))")


#-------------------------------------------------#
#   Plot the analytic and numerical dispersions   #
#-------------------------------------------------#

function plot_diatomic_dispersions(qs, numerics, analytics; savepath=nothing)
	p = plot(qs, numerics[1]; label="LA (Phunny)", lw=2, color=:blue,
		 xlabel="qx", ylabel="Energy", legend=:bottomright, framestyle=:box)
	plot!(p, qs, analytics[1]; label="LA (Analytic)", lw=2, ls=:dash, color=:black)
	plot!(p, qs, numerics[2]; label="LO (Phunny)", lw=2, color=:red)
	plot!(p, qs, analytics[2]; label="LO (Analytic)", lw=2, ls=:dash, color=:black)
	title!(p, "1D Diatomic Chain: Cu-O Dispersion")
	if isnothing(savepath)
		display(p)
	else
		savefig(p, savepath)
		@info "Saved diatomic chain dispersion plot to $(savepath)"
	end
	return p
end
spath = "~/Downloads/temporary-figures/phunny-1D-diatomic-chain.png"
plot_diatomic_dispersions(qs, [num_LA, num_LO], [ana_LA, ana_LO]; savepath=spath)


