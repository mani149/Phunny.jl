"""
This script validates the numerical results obtained using Phunny.jl
by comparing the numerical FCM to the analytic FCM for a diamond lattice.

The analytic solution may be found in Michael P. Marder's Condensed Matter
Physics textbook, for reference. 
"""
#-----------------#
# Import Packages #
#-----------------#
using LinearAlgebra, StaticArrays, Sunny, .Phunny
using Combinatorics

#--------------------------------------------------------------------#
# Diamond ~ Face-Centered Cubic Crystal				     #
#								     #
# Conventional Cell --> Basis Positions : 			     #
# 					  (0,0,0) & (1/4,1/4,1/4)    #
#--------------------------------------------------------------------#
a = 5.43; L = lattice_vectors(a,a,a,90,90,90) #Conventional Cubic
fpos = [@SVector[0.0, 0.0, 0.0], @SVector[0.25, 0.25, 0.25]]; types = ["Si", "Si"] 
cryst = Crystal(L, fpos; types=types)

#---Build Model---#
model = build_model(cryst; cutoff=0.45a, kL=10.0, kT=0.0) #Take kL = 10.0 & kT = 0.0 for simplicity
FCMs = assemble_force_constants!(model)
enforce_asr!(FCMs, model.N)

#---Physical Units---#
M = mass_vector(model) #amu
hbar = Phunny.HBAR_meV_ps

#------------------------------------------------------#
# make_physical!() ~ returns mass-weighted eigenvector #
#------------------------------------------------------#
@inline make_physical!(V,v,m,s) = SVector{3}(V[3(s-1)+1,v], V[3(s-1)+2,v], V[3(s-1)+3,v])/sqrt(m[s])

#-----------------------------------------------------------#
# longitudinal_weight!() ~ computes longitudinal phonon     #
# projection using mass-weighted eigenvectors & unit bond   #
# direction (nhat).					    #
#-----------------------------------------------------------#
@inline longitudinal_weight!(V,v,m,s,nhat) = abs2(dot(nhat, make_physical!(V,v,m,s)))




#-------------------------------------------------------#
# Extract energies and polarization weights at q-point  #
# and label by LA/TA/LO/TO using the total longitudinal #
# weight, summed over atoms. 			        #
#-------------------------------------------------------#
function branch_labels(qcart::SVector{3,Float64})
	E, V = phonons(model, FCMs, qcart; q_basis=:cart, cryst=cryst)
	nhat = norm(qcart) > 0 ? qcart/norm(qcart) : @SVector[0.0, 0.0, 1.0]
	
	wL = zeros(length(E))
	for v in 1:length(E)
		acc = 0.0
		for s in 1:model.N
			u = make_physical!(V, v, M, s)
			acc += abs2(dot(nhat,u))
		end
		wL[v] = acc
	end
	#Sort by energy, keeping weights for labels
	perm = sortperm(E)
	return E[perm], V[:,perm], wL[perm], perm
end

print("\n\n--- Begin Sanity Checks: Phonon Modes & Polarizations ---\n\n")

#-----Gamma Point Checks-----#
Eg, Vg, wLg, perm_g = branch_labels(@SVector[0.0, 0.0, 0.0])

#Expected Results: 
#	3 Acoustic Modes ~ 0 
#	3 Degenerate Optical Modes at Gamma 
#	no LO-TO splitting without Coulomb interactions.
Eg_sorted = sort(Eg); Eg_acoustic = Eg_sorted[1:3]; Eg_optical = Eg_sorted[4:6]
print("Gamma-Point Energies (meV): | \n")
print("----------------------------'\n")
print("                 Acoustic ~ $(Eg_acoustic)\n")
print("                 Optical ~ $(Eg_optical)\n\n")

#Degeneracy check
acoustic_spread = maximum(Eg_acoustic) - minimum(Eg_acoustic)
optical_spread = maximum(Eg_optical) - minimum(Eg_optical)
print("Gamma-point acoustic spread: $(acoustic_spread)\n")
print("Gamma-point optical spread: $(optical_spread)\n")
print("\n--- End Sanity Checks: Phonon Modes & Polarizationz ---\n")
print("\n|============================================================|\n")
#===============================================#
# Sanity checks for dynamical_gradient!()       #
#===============================================#
print("\n--- Begin Sanity Checks: dynamical_gradient!() ---\n")

# pick a small but finite q to avoid Γ=0 issues
q_test = 1e-2 * @SVector [1.0, 0.0, 0.0]
δq     = 2e-2
nhat   = q_test / norm(q_test)

# preallocate gradient matrices (3N×3N)
N3  = 3*model.N
dDx = zeros(ComplexF64, N3, N3)
dDy = similar(dDx)
dDz = similar(dDx)

# fill them in-place
dynamical_gradient!(dDx, dDy, dDz, model, FCMs, q_test)

# (1) Hermiticity check
# Expect ≲ 1e-12; anything >1e-8 suggests a bug or too-large δq in subsequent FD test.
herm_x = norm(Matrix(dDx) - Matrix(dDx)') / max(norm(Matrix(dDx)), 1e-16)
herm_y = norm(Matrix(dDy) - Matrix(dDy)') / max(norm(Matrix(dDy)), 1e-16)
herm_z = norm(Matrix(dDz) - Matrix(dDz)') / max(norm(Matrix(dDz)), 1e-16)
@info "Hermiticity ratios (‖A - A'‖ / ‖A‖)" herm_x herm_y herm_z
print("\n")

# (2) Finite-difference validation on D itself
# Try to see ≲ 1e-6–1e-8. If worse, reduce δq (e.g. 1e-2→5e-3) or check units.
Dp = Phunny.dynamical_matrix(model, FCMs, q_test + δq*nhat)
Dm = Phunny.dynamical_matrix(model, FCMs, q_test - δq*nhat)
dD_fd = (Matrix(Dp) - Matrix(Dm)) / (2δq)
dD_dir = nhat[1]*Matrix(dDx) .+ nhat[2]*Matrix(dDy) .+ nhat[3]*Matrix(dDz)
fd_ratio = norm(dD_dir - dD_fd) / max(norm(dD_dir), 1e-16)
@info "Finite-difference vs analytic gradient" ratio = fd_ratio
print("\n")

# (3) Γ-point acoustic numerator ~ 0 (do NOT divide by ω at Γ)
# Expect machine-small ~ 1e-12 or less.
qΓ = @SVector [0.0, 0.0, 0.0]
fill!(dDx, 0); fill!(dDy, 0); fill!(dDz, 0)
dynamical_gradient!(dDx, dDy, dDz, model, FCMs, qΓ)
dDΓ = Matrix(dDx)  # any direction equivalent for this numerator check at Γ

EΓ, VΓ = phonons(model, FCMs, qΓ; q_basis=:cart, cryst=cryst)
idx_ac = sortperm(EΓ)[1:3]  # acoustic triplet
numΓ = [ real(dot(view(VΓ,:,ν), dDΓ * view(VΓ,:,ν))) for ν in idx_ac ]
@info "Γ-point acoustic numerators (should be ≈ 0)" numΓ

print("\n--- End Sanity Checks: dynamical_gradient!() ---\n")
print("\n|============================================================|\n")
#===============================================#
#    Sanity checks for dynamical_hessian!()     #
#===============================================#

#Note, dynamical_hessian!() may be used to compute the
#Γ-point acoustic phonon mode group velocities. 
print("\n--- Begin Sanity Checks: dynamical_hessian!() ---\n")
# Preallocate once
N3 = 3*model.N
Hn = zeros(ComplexF64, N3, N3)

# Choose a direction
nh = @SVector[1.0, 1.0, 1.0]  # e.g., L

#Analytic Solver
dynamical_hessian!(Hn, model, FCMs, nh; backend=:analytic)

# Project into Γ acoustic subspace
EΓ, VΓ = phonons(model, FCMs, @SVector[0.0,0.0,0.0]; q_basis=:cart, cryst=cryst)
idx_ac  = sortperm(EΓ)[1:3]
Eac     = Matrix(view(VΓ, :, idx_ac))               # 3N×3
Hac     = Hermitian(Eac' * Hn * Eac)
λ       = sort(eigvals(Hac))                        # internal curvatures ≥ 0

#Check Hermiticity and Positive Semidefiniteness
eigvals_H = eigvals(Hac); tol = 1e-10
@assert maximum(abs.(eigvals_H .- real.(eigvals_H))) < 1e-12  # numerically Hermitian
@assert minimum(eigvals_H) > -tol                             # Positive Semidefinite within tolerance

# Convert to speeds (Å/ps)
bridge  = Phunny.ALPHA_meV / Phunny.HBAR_meV_ps
v1       = bridge .* sqrt.(max.(λ, 0.0))

#Complex-Step Solver
dynamical_hessian!(Hn, model, FCMs, nh; backend=:complexstep)

# Project into Γ acoustic subspace
EΓ2, VΓ2 = phonons(model, FCMs, @SVector[0.0,0.0,0.0]; q_basis=:cart, cryst=cryst)
idx_ac2  = sortperm(EΓ2)[1:3]
Eac2     = Matrix(view(VΓ2, :, idx_ac2))               # 3N×3
Hac2     = Hermitian(Eac2' * Hn * Eac2)
λ2       = sort(eigvals(Hac2))                        # internal curvatures ≥ 0

#Check Hermiticity and Positive Semidefiniteness
eigvals_H2 = eigvals(Hac2); tol = 1e-10
@assert maximum(abs.(eigvals_H2 .- real.(eigvals_H2))) < 1e-12  # numerically Hermitian
@assert minimum(eigvals_H2) > -tol                              # Positive Semidefinite within tolerance


# Convert to speeds (Å/ps)
v2       = bridge .* sqrt.(max.(λ2, 0.0))
print("\nAnalytic and Complex-Step Solution Equal? $((x->round.(x))(v1) == (x->round.(x))(v2))!\n")
print("\n")
@show v1
@show v2
print("\n")

#Compute Γ-point acoustic mode group velocities 
#along high-symmetry axes using the dynamic Hessian
for (lbl, dir) in [("100", @SVector[1,0,0]),
                   ("110", @SVector[1,1,0]),
                   ("111", @SVector[1,1,1])]
    nh3 = dir / norm(dir)
    fill!(Hn, 0); dynamical_hessian!(Hn, model, FCMs, nh3; backend=:analytic)
    EΓ3, VΓ3 = phonons(model, FCMs, @SVector[0.0,0.0,0.0]; q_basis=:cart, cryst=cryst)
    idx = sortperm(EΓ3)[1:3]
    Eac3 = Matrix(view(VΓ3, :, idx))
    Hac3 = Hermitian(Eac3' * Hn * Eac3)
    λ3   = sort(eigvals(Hac3))
    v3   = (Phunny.ALPHA_meV / Phunny.HBAR_meV_ps) .* sqrt.(max.(λ3,0.0))
    deg = abs(v3[2]-v3[1]) / max(1e-12, max(abs(v3[1]),abs(v3[2])))
    print("[$(lbl)] Γ (Hessian): v = [$(v3[1]), $(v3[2]), $(v3[3])] Å/ps | TA-deg=$(deg)\n")
end
print("\n--- End Sanity Checks: dynamical_hessian!() ---\n")


#==========================================#
# Compare with analytic solution in Marder #
#==========================================#




#=
#################################################################
#This section is unfinished and will be fixed later.		#
#								#
#It is unnecessary to include for the simple proof-of-principle #
#analysis that this script is attempting to perform.        # # #
#==========================================================#
# Compute Finite-q Group Velocities using Dynamical Matrix #
#==========================================================#
# ------------------------------------------------------#
# In-place, buffer-reusing version (optimal for sweeps)	#
# ------------------------------------------------------#
"""
compute_group_velocities!(v, Ddx,Ddy,Ddz, model, Φ, q_cart, nhat; cryst)
Fill v (length 3N) with velocities (Å/ps). Ddx/Ddy/Ddz are 3N×3N ComplexF64 buffers.
"""
function compute_group_velocities!(v::AbstractVector{Float64},
                                   dDx::AbstractMatrix{ComplexF64},
                                   dDy::AbstractMatrix{ComplexF64},
                                   dDz::AbstractMatrix{ComplexF64},
                                   model::Model,
                                   Φ::Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3,Float64,9}},
                                   q_cart::SVector{3,Float64},
                                   nhat::SVector{3,Float64}; cryst=cryst)

    nh = nhat / norm(nhat)

    # eigenpairs
    EmeV, V = phonons(model, Φ, q_cart; q_basis=:cart, cryst=cryst) #meV
    
    #Physical eigenvalues
    ω = EmeV ./ Phunny.HBAR_meV_ps # 1/ps
    
    #Physical unit scaling
    scaling = (Phunny.ALPHA_meV / Phunny.HBAR_meV_ps)^2 # 1/(ps)^2

    # gradients into preallocated buffers
    dynamical_gradient!(dDx, dDy, dDz, model, Φ, q_cart)

    # dD along nhat (use a temporary view to avoid full new alloc if desired)
    dD = nh[1]*dDx .+ nh[2]*dDy .+ nh[3]*dDz

    @inbounds for ν in 1:length(EmeV)
        if ω[ν] > 0
            eν = view(V, :, ν)
            num = real(dot(eν, dD * eν))
            v[ν] = (num / (2*ω[ν]))*scaling # Å/ps = Å(1/(ps)^2)/(1/ps)
        else
            v[ν] = 0.0
        end
    end
    return v
end


#===========================================================#
# Γ-limit group velocities using in-place buffer reusing    #
# function: compute_group_velocities!()			    #
#===========================================================#
print("\n--- Begin Sanity Checks: compute_group_velocity!() ---\n")
@info "Γ-limit velocities computed using ∂D/∂q\n"
print("\n")
# Preallocate reusable matrices and velocity array
N3 = 3 * model.N
dDx = zeros(ComplexF64, N3, N3)
dDy = similar(dDx)
dDz = similar(dDx)
v   = zeros(Float64, N3)

# helper for neat directional printout
function report_dir(label::String, dir::SVector{3,Float64})
    qmag = 1e-2
    q = qmag * dir / norm(dir)
    nhat = dir / norm(dir)

    # compute velocities in-place
    compute_group_velocities!(v, dDx, dDy, dDz, model, FCMs, q, nhat; cryst=cryst)

    # label LA/TA within acoustic triplet
    E, V = phonons(model, FCMs, q; q_basis=:cart, cryst=cryst)
    idx_ac = sortperm(E)[1:3]
    M = model.mass
    wL = [@inbounds sum(longitudinal_weight!(V,ν,M,s,nhat) for s in 1:model.N)
          for ν in 1:length(E)]
    la = idx_ac[argmax(wL[idx_ac])]
    T  = filter(i->i!=la, idx_ac)
    vL, vT1, vT2 = v[la], v[T[1]], v[T[2]]

    # degeneracy measure for TA modes
    deg = abs(vT1 - vT2) / max(1e-12, max(abs(vT1), abs(vT2)))

    print("[$(label)] | vL = $(vL) Å/ps | vT = $(vT1), $(vT2) | degeneracy = $(deg)\n")
end

report_dir("100", @SVector[1.0, 0.0, 0.0])
report_dir("110", @SVector[1.0, 1.0, 0.0])
report_dir("111", @SVector[1.0, 1.0, 1.0])

print("\n--- End Sanity Checks: compute_group_velocity!() ---\n")
=#


