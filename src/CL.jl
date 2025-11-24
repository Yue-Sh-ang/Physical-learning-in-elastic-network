module TrainerCL

using Random
using LinearAlgebra
using .ENM


mutable struct Trainer_CL
    input::Vector{Tuple{Int,Float64,Float64}}   # (edge, input strain, stiffness)
    output::Vector{Tuple{Int,Float64,Float64}} # (edge, target strain, stiffness)
    trainmask::Vector{Bool}
    net_f::ENM
    net_c::ENM
end

function Trainer_CL(net::ENM,
                   input::Vector{Tuple{Int,Float64,Float64}},
                   output::Vector{Tuple{Int,Float64,Float64}})

    trainmask = trues(net.ne)
    net_f = net
    net_c = deepcopy(net)

    # modify training masks + rest lengths + stiffnesses
    for (edge, strain, stiff) in input
        trainmask[edge] = false
        net_f.l0[edge] *= (1 + strain)
        net_c.l0[edge] *= (1 + strain)
        net_f.k[edge] = stiff
        net_c.k[edge] = stiff
    end

    for (edge, _, _) in output
        trainmask[edge] = false
    end

    return Trainer_CL(input, output, trainmask, net_f, net_c)
end

function calc_strain_f(tr::Trainer_CL, edge::Int)
    (u, v) = tr.net_f.edges[edge]
    dx = tr.net_f.pts[v, :] .- tr.net_f.pts[u, :]
    dist = norm(dx)
    l0 = tr.net_f.l0[edge]
    return (dist - l0) / l0
end

function clamp_eta!(tr::Trainer_CL,eta)
    for (edge, strain_t, stiff) in tr.output
        strain_f = calc_strain_f(tr, edge)
        strain_c = strain_f + (strain_t - strain_f)*eta

        tr.net_c.l0[edge] *= (1 + strain_c)
        tr.net_c.k[edge] = stiff
    end
end

function learn_k!(tr::Trainer_CL, alpha::Float64;dt=0.005)
    G = zeros(length(tr.net_f.k))

    @inbounds for (ide, (u,v)) in enumerate(tr.net_f.edges)
        if !tr.trainmask[ide]
            continue
        end

        # forward
        lf = norm(tr.net_f.pts[v,:] .- tr.net_f.pts[u,:])
        l0f = tr.net_f.l0[ide]

        # clamped
        lc = norm(tr.net_c.pts[v,:] .- tr.net_c.pts[u,:])
        l0c = tr.net_c.l0[ide]

        G[ide] =  (lc - l0c)^2 - (lf - l0f)^2
    end

    tr.net_f.k .-= dt*alpha .* G
    tr.net_c.k .-= dt*alpha .* G
end

function step!(tr::Trainer_CL,T; eta=1.0, alpha=1.0, step_md=1)
    seed = rand(1:10^6)

    # forward dynamics
    run_md!(tr.net_f,T; steps=step_md, seed=seed)

    # clamp
    clamp_eta!(tr; eta=eta)

    # constrained dynamics
    run_md!(tr.net_c,T; steps=step_md, seed=seed)

    # stiffness update
    learn_k!(tr, alpha)

    # return list of strains on output edges
    return [calc_strain_f(tr, edge) for (edge, _, _) in tr.output]
end
