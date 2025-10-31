using LinearAlgebra, StaticArrays, Sunny, .Phunny, GLMakie, Statistics

#---------------------------------------------------------------#
#References: 							#
#	     - Jorgensen et al., Phys. Rev. B 38, 11337 (1988) 	#
#	     - Radaelli et al., Phys. Rev. B 49, 4163 (1994) 	#
#---------------------------------------------------------------#



################
# Define Model #
################


#Lattice Geometry : Low Temperature, Space Group: Bmab
a,b,c = 5.35/sqrt(2), 5.41/sqrt(2), 13.2 
L = lattice_vectors(a, b, c, 90, 90, 90)

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

#Crystal + Model
cryst = Crystal(L, fpos; types) 
cutoff = 0.8*minimum((a,b,c)); 
model = build_model(cryst; cutoff)

#Manually Defined Bonds & Force Constants
bond_dict = Dict( (:Cu, :O) => (190.0, 28.0),
		  (:O,  :O) => (40.00, 4.00),
		  (:O,  :La) => (20.00, 1.00))
#Index atoms
atoms = atomic_index(types)

#Mass Sanity Check
@info "Mass (Min, Max) : $(extrema(model.mass))" 


############
# Analysis #
############

#Calibration
for (kL_CuO, kT_CuO) in ((1,1), (10,1), (33,2), (50,5), (190,28), (210,32), (220, 35))
    bond_dict[:Cu,:O] = (kL_CuO, kT_CuO)
    
    assign_force_constants!(model, atoms, bond_dict)
    FCMs = assemble_force_constants!(model; β_bend=0.2, bend_shell=:nn, bend_tol=0.15); enforce_asr!(FCMs, model.N)

    EΓ, _ = phonons(model, FCMs, @SVector[0.0, 0.0, 0.0]; 
                    q_basis=:rlu, q_cell=:primitive, cryst=cryst)
    @info "Cu-O (kL=$(kL_CuO)) ⟶ Γ optics ≈  $(EΓ[4:end])"
end


#################
# FCM & Phonons #
#################
bond_dict[:Cu,:O] = (190, 28)
assign_force_constants!(model, atoms, bond_dict)

#Assemble Force Constants Matrices
ϕ = assemble_force_constants!(model; β_bend=0.2, bend_shell=:nn, bend_tol=0.15)
enforce_asr!(ϕ, model.N)

#Minimum Image
@inline minimg(x) = x .- round.(x)

#Convert Fractional -> Cartesian (Real-Space)
@inline r_cartesian(Lat, r1, r2) = norm(Lat*minimg(r2 - r1))

planar_x = r_cartesian(L, fpos[1], fpos[2])
planar_y = r_cartesian(L, fpos[1], fpos[3])
apical = r_cartesian(L, fpos[1], fpos[4])

@assert 1.85 ≤ planar_x ≤ 1.95 && 1.85 ≤ planar_y ≤ 1.95 "Planar oxygen bond distance does NOT match the experimental values!"
@assert 2.30 ≤ apical ≤ 2.45 "Apical oxygen bond distance does NOT match the experimental values!"

#DSF-based check: stable phonons & sensible optical energies at Γ-point
eigenpairs = phonons(model, ϕ, @SVector[0.0, 0.0, 0.0]; 
		     q_basis=:rlu, q_cell=:primitive, cryst=cryst)

@assert any(ω -> 60.0 ≤ ω ≤ 95.0, eigenpairs[1]) "Expected oxygen-dominant optical modes in ~[60, 95] meV window for La2CuO4!"



############
# Plotting #
############


#Plot Phonon DoS
hist(eigenpairs[1], bins=40, color="royalblue", label="Phonon DoS")


#Plots S(q,ω) = ∑ₙ S(qₙ, ω) : S(qₙ,ω) = S[qₙ,:]
function plot_dsf_line!(cryst, model, Φ; 
                        q₀=@SVector[0.0, 0.0, 0.0], 
                        q₁=@SVector[1.5, 0.0, 0.0], 
                        nq=81, ωmax=400.0, nω=1201, 
                        η=1.0, q_cell=:primitive)
    qs = [SVector{3,Float64}((1-t).*q₀ .+ t.*q₁) for t in range(0,1;length=nq)]
    ωs = range(0.0, ωmax; length=nω)
    σ = η/sqrt(8*log(2))

    Sqω = zeros(Float64, nq, nω)
    @inbounds for (iq, q) in enumerate(qs)
        Sω = onephonon_dsf(model, Φ, q, ωs; q_basis=:rlu, q_cell=q_cell, cryst=cryst, T=0.25)
        Sqω[iq,:] .= Sω
    end
    
    @inline lohi(z) = begin
	lo = 0.0
	m = (isfinite.(z) .& ( z .> 0.0))
	hi = any(m) ? mean(z[m]) : maximum(z)/2
	(lo, hi)
    end

    fig = Figure(size=(500,500))
    ax = Axis(fig[1,1], xlabel="q index (Γ ↦ X)", ylabel = "Energy (meV)", title="La₂CuO₄ | One-Phonon S(q,ω)") 
    hm = heatmap!(ax, 1:nq, ωs, Sqω ; interpolate=true, colormap=:viridis, colorrange=lohi(Sqω))
    Colorbar(fig[1,2], hm; label="Intensity")
    screen=display(fig); wait(screen)
end
plot_dsf_line!(cryst, model, ϕ)




