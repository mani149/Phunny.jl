# --- bond class key -----------------------------------------------------------
# species: sorted pair, shell: neighbor shell index (1,2,...),
# orient: :planar | :apical | :other (for layered oxides)
struct BondClass
    species::NTuple{2,Symbol}
    shell::Int
    orient::Symbol
end
Base.:(==)(a::BondClass,b::BondClass) = a.species==b.species && a.shell==b.shell && a.orient==b.orient
Base.hash(b::BondClass,h::UInt) = hash(b.species, hash(b.shell, hash(b.orient,h)))

# --- classify bonds present in the model -------------------------------------
#
#Return:
# - classes::Dict{BondClass,Vector{Int}} mapping each class to model.edge indices
# - clist::Vector{BondClass} stable ordering of classes
#
function bond_classes(model; c_axis::SVector{3,Float64}=@SVector[0.0,0.0,1.0], planar_tol::Float64=0.25)
    # first pass: collect distances by (species-pair, orient)
    bykey = Dict{Tuple{NTuple{2,Symbol},Symbol}, Vector{Tuple{Int,Float64}}}() # (bond_index, r)
    spair(i,j) = begin
        s1 = Symbol(model.species[i]); s2 = Symbol(model.species[j])
        (min(s1,s2), max(s1,s2))
    end
    for (e,b) in enumerate(model.bonds)
        r̂ = b.r0 / norm(b.r0)
        c = abs(dot(r̂, c_axis))
        orient = c < planar_tol ? :planar : (c > 1 - planar_tol ? :apical : :other)
        key = (spair(b.i, b.j), orient)
        push!(get!(bykey, key, Vector{Tuple{Int,Float64}}()), (e, norm(b.r0)))
    end
    # define shells per (species,orient) by unique distances
    classes = Dict{BondClass, Vector{Int}}()
    clist = BondClass[]
    for (key, items) in bykey
        # unique distances with a small tolerance; nearest-neighbor clusters → shell 1,2,...
        ds = sort!(unique(round.(last.(items), digits=3)))  # 10^-3 Å binning
        for (e, r) in items
            shell = findfirst(d -> isapprox(r, d; atol=1e-3), ds)::Int
            bc = BondClass(key[1], shell, key[2])
            if !haskey(classes, bc); classes[bc] = Int[]; push!(clist, bc); end
            push!(classes[bc], e)
        end
    end
    return classes, clist
end

# --- seeding heuristics (no user input needed) --------------------------------
#
#Seed (kL,kT) per bond class using geometry:
# kL ∝ 1/r^3 with species-dependent prefactor; kT = γ*kL with γ by orient.
#Returns Vector θ of length 2*nc storing log-parameters to enforce positivity: k = exp(θ).
#
function seed_params(model, classes, clist; α=Dict{Tuple{Symbol,Symbol},Float64}(),
                     γ_planar=0.15, γ_apical=0.05, γ_other=0.10)
    default_α(sp) = get(α, sp, 50.0)
    rrep = Dict{BondClass,Float64}()
    for (key, idxs) in classes
        rs = map(e -> norm(model.bonds[e].r0), idxs)
        rrep[key] = sum(rs)/length(rs)
    end
    θ = zeros(2*length(clist))
    for (k, key) in enumerate(clist)
        αsp = default_α(key.species); r0 = rrep[key]
        kL = αsp / (r0^3)
        γ  = key.orient===:planar ? γ_planar : key.orient===:apical ? γ_apical : γ_other
        θ[2k-1] = log(kL + eps()); θ[2k] = log(γ*kL + eps())
    end
    return θ
end

# --- inject per-class parameters into the model FCM ---------------------------
#
#Write kL,kT from θ into all edges of each class, then assemble FCMs.
#Calls enforce_asr! to satisfy translational invariance.
#
function update_fcm!(θ, model, classes, clist)
    @inbounds for (k, key) in enumerate(clist)
        kL = exp(θ[2k-1]); kT = exp(θ[2k])
        for e in classes[key]
            b = model.bonds[e]
            model.bonds[e] = Bond(b.i, b.j, b.R, b.r0, kL, kT)  # Bond is immutable → replace entry
        end
    end
    Φ = assemble_force_constants!(model)
    enforce_asr!(Φ, model.N)
    return Φ
end

# --- target definitions (optional light fit) ----------------------------------
# Minimal targets struct; you can extend as needed.
struct AutoTargets
    # acoustic target: longitudinal & transverse sound speeds along a few dirs near Γ
    v_targets::Vector{Tuple{SVector{3,Float64}, SVector{3,Float64}}} # (qhat, v_L, v_T) with v in m/s
    # optic energies (meV) at high-symmetry points (Γ, X, M, etc.)
    optic_targets::Vector{Tuple{SVector{3,Float64}, Vector{Float64}}} # (kpt, ωlist)
end

# Compute residuals given θ (small and cheap; D(q) linear in FCs)
function residuals(θ, model, classes, clist, cell, targets::AutoTargets)
    FCMs = update_fcm!(θ, model, classes, clist)
    r = Float64[]
    # acoustic slopes at tiny q
    for (qhat, vLVT) in targets.v_targets
        vL_tar, vT_tar = vLVT[1], vLVT[2]
        vL_num, vT_num = acoustic_velocities_from_FC(FCMs, cell; qhat)  # you have/plan helpers for this
        push!(r, (vL_num - vL_tar))
        push!(r, (vT_num - vT_tar))
    end
    # optic energies
    for (kpt, ωtar) in targets.optic_targets
        ωnum, _ = phonons(model, FCMs, kpt; q_basis=:rlu, q_cell=:primitive, cryst=cell)  # returns all branches (meV)
        # take a stable subset: sort and match shortest diffs
        ωn = sort(ωnum); ωt = sort(ωtar)
        # isolate optical modes
        ωopt = @view ωn[4:end]
        # match each target to its nearest optic mode (robust if counts differ)
        for t in sort(ωtar)
            j = findmin(abs.(ωopt .- t))[2]
            push!(r, ωopt[j] - t)
        end
        #m = min(length(ωn), length(ωt))
        #append!(r, @view(ωn[1:m]) .- @view(ωt[1:m]))
    end
    return r
end

# Finite-difference Jacobian (small problems; robust)
function jacobian!(J, θ, model, classes, clist, cell, targets; epsθ=1e-3)
    r0 = residuals(θ, model, classes, clist, cell, targets)
    m = length(r0); n = length(θ)
    #resize!(J, m, n)
    J = Matrix{Float64}(undef, m, n)
    for j in 1:n
        θp = copy(θ); θp[j] += max(0.05, 0.1*abs(θ[j])) # relative step in log-k (~5% - 10% in k)
        rp = residuals(θp, model, classes, clist, cell, targets)
        @inbounds @views J[:,j] .= (rp .- r0) ./ epsθ
    end
    return r0, J
end

# --- Gauss–Newton with damping (no extra deps) --------------------------------
#"""
#Fit per-class (kL,kT) to supplied targets. If no targets are given, this returns seeded values.
#Regularization λ discourages extreme k ratios.
#"""
function fit_per_pair!(model, cell; targets::Union{Nothing,AutoTargets}=nothing,
                       planar_tol=0.25, λ=1e-4, maxiter=60)
    classes, clist = bond_classes(model; planar_tol)
    θ = seed_params(model, classes, clist)
    if targets === nothing
        FCMs = update_fcm!(θ, model, classes, clist)
        return FCMs, Dict(clist .=> [(exp(θ[2i-1]), exp(θ[2i])) for i in 1:length(clist)])
    end
    J = Matrix{Float64}(undef, 0, 0)
    for it in 1:maxiter
        r0, J = jacobian!(J, θ, model, classes, clist, cell, targets)
        # Tikhonov regularization on θ (keeps kL,kT moderate)
        A = J'J + λ*I
        b = -J'r0
        Δθ = A \ b
        θ_trial = θ .+ Δθ
        r1, _ = residuals(θ_trial, model, classes, clist, cell, targets)
        F0 = 0.5*dot(r0,r0) + 0.5λ*dot(θ,θ)
        F1 = 0.5*dot(r1,r1) + 0.5λ*dot(θ_trial,θ_trial)

        # backtrack up to 4 times if not improving
        tries = 0
        while F1 > F0 && tries < 4
            Δθ .*= 0.5
            θ_trial .= θ .+ Δθ
            r1, _ = residuals(θ_trial, model, classes, clist, cell, targets)
            F1 = 0.5*dot(r1,r1) + 0.5λ*dot(θ_trial,θ_trial)
            tries += 1
        end
        
        # simple floor to avoid vanishing k (bond-agnostic clamp)
        θmin = log(0.5)   # k ≥ 0.5 (units whatever your internal N/m equivalent is)
        @inbounds for j in eachindex(θ_trial)
            θ_trial[j] = max(θ_trial[j], θmin)
        end
        
        if norm(Δθ) < 1e-4
            θ = θ_trial
            break
        end
        θ = θ_trial
    end
    FCMs = update_fcm!(θ, model, classes, clist)
    params = Dict{BondClass,Tuple{Float64,Float64}}()
    for (i,key) in enumerate(clist)
        params[key] = (exp(θ[2i-1]), exp(θ[2i]))
    end
    return FCMs, params
end
