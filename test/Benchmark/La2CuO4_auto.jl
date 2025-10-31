using LinearAlgebra, StaticArrays, Sunny, .Phunny, GLMakie, Statistics

#---------------------------------------------------------------#
#References: 							#
#	     - Jorgensen et al., Phys. Rev. B 38, 11337 (1988) 	#
#	     - Radaelli et al., Phys. Rev. B 49, 4163 (1994) 	#
#---------------------------------------------------------------#


#Lattice Geometry
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

#This function is included in the core Phunny API
@inline atomic_index(labels) = Dict(i => Symbol(labels[i]) for i in eachindex(labels))

################################ To Do: Separate this script into two files! 
# THIS SECTION IS EXPERIMENTAL #                                (1) Manually Defined Force Constants ~ Empirical Fit
################################                                (2) Automatically Defined Force Constants ~ Inferred Fit 
function _local_axis(model::Model, i::Int; kmax=8)
    #k-nearest neighbor directions around atom i (by |r₀|)
    vs = SVector{3,Float64}[]
    for b in model.bonds
        if b.i == i; push!(vs, b.r0/norm(b.r0))
        elseif b.j == i; push!(vs, -b.r0/norm(b.r0)); end
        length(vs) ≥ kmax && break
    end
    length(vs) < 2 && return @SVector[0.0, 0.0, 1.0] #fallback
    C = zero(SMatrix{3,3,Float64,9}); for v in vs; C += v*v'; end
    #principle axis = eigenvector with smallest eigenvalue (normal to local plane)
    vals, vecs = eigen(Symmetric(C))
    return SVector{3,Float64}(vecs[:,argmin(vals)]) #plane normal
end

@inline function _orient_axis(model::Model, b::Phunny.Bond; tol=0.25)
    n = _local_axis(model, b.i)
    c = abs(dot(n, b.r0 / norm(b.r0)))
    c > 1 - tol ? :apical : (c < tol ? :planar : :other)
end

function _reduced_mass(model::Model, i::Int, j::Int)
    mi = model.mass[i]; mj = model.mass[j]
    return (mi*mj)/(mi+mj)
end

function infer_force_constants!(model::Model, atom::Dict{Int,Symbol}; p::Float64=3.0, 
                                  base_pair::Tuple{Symbol,Symbol}=(:Cu,:O), 
                                  orient_factor=Dict(:planar=>1.0, :apical=>0.6, :other=>0.8), 
                                  fracT=Dict{Tuple{Symbol,Symbol},Float64}(),  
                                  pair_override=Dict{Tuple{Symbol,Symbol},Float64}(),
                                  target::Union{Nothing, Float64}=nothing, cryst=nothing)
    #atom = atomic_index(types)
    @inline canonical!(s1,s2) = s1 <= s2 ? (atom[s1], atom[s2]) : (atom[s2], atom[s1])
    #per-pair distances and reduced masses
    pair_r = Dict{Tuple{Symbol,Symbol},Vector{Float64}}()
    pair_μ = Dict{Tuple{Symbol, Symbol},Vector{Float64}}()
    for b in model.bonds
        sp = canonical!(b.i, b.j)
        push!(get!(pair_r, sp, Float64[]), norm(b.r0))
        push!(get!(pair_μ, sp, Float64[]), _reduced_mass(model, b.i, b.j))
    end
    
    #reference per-pair median
    rref = Dict(sp => median(rs) for (sp, rs) in pair_r)
    μref = Dict(sp => median(ms) for (sp, ms) in pair_μ)
    
    #Define base pair
    μbase = haskey(μref, base_pair) ? μref[base_pair] : median(values(μref))

    #S_pair from μ ratio (unless overridden)
    S_pair = Dict{Tuple{Symbol,Symbol},Float64}()
    for sp in keys(rref)
        S_pair[sp] = get(pair_override, sp, get(μref, sp, μbase)/μbase)
    end

    #k0 scale: eiter from Γ optic target for base pair or 1.0
    k0 = 1.0
    if target !== nothing && cryst !== nothing
        #rough single-step estimate: k ~ (E/α)^2 μ
        k0 = (target / Phunny.ALPHA_meV)^2 * μbase #/ S_pair[base_pair]
    end

    #apply per-bond
    for e in eachindex(model.bonds)
        b = model.bonds[e]; bmag = norm(b.r0)
        sp = canonical!(b.i, b.j)
        orientation = _orient_axis(model, b)
        factor = get(orient_factor, orientation, 1.0)
        rr = get(rref, sp, bmag)
        kL = k0*S_pair[sp]*factor*(rr/bmag)^p
        kT = get(fracT, sp, 0.10)*kL
        model.bonds[e] = Phunny.Bond(b.i, b.j, b.R, b.r0, kL, kT)
    end
    return model
end

# rescale all springs by s
function _rescale_all_bonds!(model::Model, s::Float64)
    @inbounds for e in eachindex(model.bonds)
        b = model.bonds[e]
        model.bonds[e] = Phunny.Bond(b.i,b.j,b.R,b.r0, s*b.kL, s*b.kT)
    end
end

# calibrate so the first optic at Γ hits `target` (meV)
function calibrate_global_scale!(model::Model, cryst; target=85.0, iters=3)
    Φ = assemble_force_constants!(model); enforce_asr!(Φ, model.N)
    for _ in 1:iters
        EΓ, _ = phonons(model, Φ, @SVector[0.0,0.0,0.0];
                        q_basis=:rlu, q_cell=:primitive, cryst=cryst)
        E = sort(EΓ)[4:end]                 # skip 3 acoustic
        m = mean(E[1:min(3,end)])           # average of first few optics
        s = (target / max(m,1e-9))^2        # E ∝ √k ⇒ k → s k
        if abs(s-1) < 0.05; break; end
        _rescale_all_bonds!(model, s)
        Φ = assemble_force_constants!(model); enforce_asr!(Φ, model.N)
    end
    return Φ
end

function plot_dsf_line!(cryst, model, Φ; 
                        q₀=@SVector[0.0, 0.0, 0.0], 
                        q₁=@SVector[0.5, 0.0, 0.0], 
                        nq=81, ωmax=420.0, nω=1201, 
                        η=0.8, q_cell=:primitive)
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
    ax = Axis(fig[1,1], xlabel="q index (Γ ↦ X)", ylabel = "Energy (meV)", title="La₂CuO₄ | One-Phonon S(q,ω) | Inferred Force Constants") 
    hm = heatmap!(ax, 1:nq, ωs, Sqω ; interpolate=true, colormap=:viridis, colorrange=lohi(Sqω))
    Colorbar(fig[1,2], hm; label="Intensity")
    screen=display(fig); wait(screen)
end

#End Function List

infer_force_constants!(model, atomic_index(types); base_pair=(:Cu,:O), target=85.0, cryst=cryst)
Φ = calibrate_global_scale!(model, cryst; target=85.0)
@info "Sanity check: "
EΓ, _ = phonons(model, Φ, @SVector[0.0, 0.0, 0.0]; q_basis=:rlu, q_cell=:primitive, cryst=cryst)
@info "Γ optics (first few): ", EΓ[4:10]
print("Expected: EΓ ~ [80, 100] meV optic oxygen doublet for target=85 meV.\n")
############################
# END EXPERIMENTAL SECTION #
############################







