
using LinearAlgebra, Random

function load_graph(filename)
####---------------------
# load graph from a text file
# format:
# n
# x1 y1 z1
# x2 y2 z2
# ...
# xn yn zn
# ne
# node1_1 node1_2 stiffness1 l0_1
# node2_1 node2_2 stiffness2 l0_2
# ...   
####---------------------
    lines = readlines(filename)
    idx = 1
    ### --- Read points ---
    n = parse(Int, lines[idx]); idx += 1
    pts = zeros(n, 3)
    for i in 1:n
        x, y, z = parse.(Float64, split(lines[idx]))
        pts[i, :] .= (x, y, z)
        idx += 1
    end

    ### --- Read edges ---
    ne = parse(Int, lines[idx]); idx += 1
    edges = Vector{Tuple{Int,Int}}(undef, ne)
    stiffness = Vector{Float64}(undef, ne)
    l0 = Vector{Float64}(undef, ne)

    for e in 1:ne
        cols = split(lines[idx])
        node1 = parse(Int, cols[1])
        node2 = parse(Int, cols[2])
        stiff = parse(Float64, cols[3])
        l0val = parse(Float64, cols[4])

        edges[e] = (node1, node2)
        stiffness[e] = stiff
        l0[e] = l0val

        idx += 1
    end

    return n, ne, pts, edges, stiffness, l0
end


struct ENM
    n::Int
    ne::Int
    pts0::Matrix{Float64}     # n × 3
    pts::Matrix{Float64}     # n × 3
    vel::Matrix{Float64}     # n × 3
    force::Matrix{Float64}   # n × 3
    edges::Vector{Tuple{Int,Int}}
    k::Vector{Float64}
    l0::Vector{Float64}
    m::Float64 #assuming uniform mass
    
end

function ENM(filename; m=1.0, T0=0.0,seed=123)
    
    
    n, ne, pts, edges,kvec, lvec = load_graph(filename)
    pts0=deepcopy(pts)
    vel   = zeros(n, 3)
    force = zeros(n, 3)

    # initial velocities
    Random.seed!(seed)
    for i in 1:n
        vel[i, :] = randn(3) * sqrt(T0/m)
    end

    return ENM(n, ne, pts0,pts, vel, force, edges, kvec, lvec,
               m)
end

function calc_elastic_force!(enm::ENM)
    fill!(enm.force, 0.0)

    @inbounds for i in 1:enm.ne
        (u,v) = enm.edges[i]

        dx = enm.pts[v, :] .- enm.pts[u, :]
        dist = norm(dx)

        if dist == 0
            continue
        end

        fmag = enm.k[i] * (dist - enm.l0[i])
        fvec = fmag / dist .* dx

        enm.force[u, :] .+= fvec
        enm.force[v, :] .-= fvec
    end
end

function reset_config!(enm::ENM)
    enm.pts .= enm.pts0
end

function run_md!(enm::ENM,T; steps=1, dt=0.005, seed=nothing, tau=1.0)
    if seed !== nothing # I think there is a better way to control the thermal noise
        Random.seed!(seed)
    end
    
    m   = enm.m
    τ   = tau
    
    sigma = (T > 0 && τ > 0) ? sqrt(2*m*T/(τ*dt)) : 0.0

    for _ in 1:steps
        
        # ---- 1st half-step ----
        calc_elastic_force!(enm)
        Ft = copy(enm.force)

        rand1 = sigma == 0 ? zeros(enm.n,3) : randn(enm.n,3)
        Flan1 = @. -(m/τ)*enm.vel + sigma*rand1

        @. enm.vel += 0.5*dt * (Ft + Flan1) / m

        # ---- position update ----
        @. enm.pts += dt * enm.vel

        # ---- 2nd half-step ----
        calc_elastic_force!(enm)
        Ftp = copy(enm.force)

        rand2 = sigma == 0 ? zeros(enm.n,3) : randn(enm.n,3)
        Flan2 = @. -(m/τ)*enm.vel + sigma*rand2

        @. enm.vel += 0.5*dt * (Ftp + Flan2) / m
    end
end

function calc_elastic_jacobian(enm::ENM)
    J = zeros(3*enm.n, 3*enm.n)

    @inbounds for i in 1:enm.ne
        (u,v) = enm.edges[i]

        dx = enm.pts[v,:] .- enm.pts[u,:]
        dist = norm(dx)
        dist == 0 && continue

        k  = enm.k[i]
        l0 = enm.l0[i]

        term1 = k*(1 - l0/dist)
        term2 = k*l0/(dist^3)

        for a in 1:3
            for b in 1:3
                val = term1*(a==b) + term2*dx[a]*dx[b]
                J[3u-2+a, 3u-2+b] += val
                J[3v-2+a, 3v-2+b] += val
                J[3u-2+a, 3v-2+b] -= val
                J[3v-2+a, 3u-2+b] -= val
            end
        end
    end
    return J
end

function calc_modes(enm::ENM)
    J = calc_elastic_jacobian(enm)
    eigen(J)
end

