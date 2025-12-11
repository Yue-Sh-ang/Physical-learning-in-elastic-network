
using Random
using LinearAlgebra

const INPUT = 1
const OUTPUT = 2
const NORMAL = 0


mutable struct Trainer_CL
    input::Vector{Tuple{Int,Float64,Float64,Float64}}   # (edge, input strain, stiffness, l0)
    output::Vector{Tuple{Int,Float64,Float64,Float64}} # (edge, target strain, stiffness,l0)
    edgetype::Vector{Int}    # 0: free, 1: input, 2: output
    net_f::ENM
    net_c::ENM
end

function Trainer_CL(net::ENM,
                   input::Vector{Tuple{Int,Float64,Float64}},
                   output::Vector{Tuple{Int,Float64,Float64}})

    edgetype = zeros(Int, net.ne)
    net_f = deepcopy(net)
    net_c = deepcopy(net)

    # modify training masks + rest lengths + stiffnesses
    input_construct = [(edge, strain, stiff, net.l0[edge]) for (edge, strain, stiff) in input]
    for (edge, strain, stiff, l0) in input_construct
        edgetype[edge] = INPUT
        put_strain!(net_f, edge, l0*(1+strain); k=stiff)
        put_strain!(net_c, edge, l0*(1+strain); k=stiff)
    end
    
    output_construct = [(edge, strain, stiff, net.l0[edge]) for (edge, strain, stiff) in output]
    for (edge, strain, stiff, l0) in output_construct
        edgetype[edge] = OUTPUT
        net_f.k[edge] = 0.0   
        net_c.k[edge] = 0.0
    end

    return Trainer_CL(input_construct, output_construct, edgetype, net_f, net_c)
end



function clamp_eta!(tr::Trainer_CL,strain_f::Vector{Float64},eta)
    
    for (ido,(edge, strain_t, stiff, l0)) in enumerate(tr.output)
        strain_c = strain_f[ido] + (strain_t - strain_f[ido])*eta
        put_strain!(tr.net_c, edge, l0*(1+strain_c); k=stiff)
    end
end


function learn_k!(tr::Trainer_CL, Gradient::Vector{Float64},alpha,vmin=1e-3, vmax=2e2)
    
    tr.net_f.k .-= alpha * Gradient
    tr.net_c.k .-= alpha * Gradient
    
    tr.net_c.k .= clamp.(tr.net_f.k, vmin, vmax) # might comflict with stiffness at input or output careful!
    tr.net_f.k .= clamp.(tr.net_f.k, vmin, vmax)
    
    
end 

function step!(tr::Trainer_CL,T;sf_old= nothing, eta=1.0, alpha=1.0, step_md=10)
    #learning information reset
    Gradient=zeros(length(tr.net_f.edges))
    sf_new = zeros(length(tr.output))

    if sf_old != nothing
        clamp_eta!(tr,sf_old,eta)
    else
        clamp_eta!(tr,zeros(length(tr.output)), eta)
    end
    

    
    
    for _ in 1:step_md
        run_md!(tr.net_f,T)
        run_md!(tr.net_c,T)
        # update gradient
        @inbounds for (ide, (u,v)) in enumerate(tr.net_f.edges)
            if tr.edgetype[ide] == NORMAL
                lf = norm(tr.net_f.pts[v,:] .- tr.net_f.pts[u,:])
                l0f = tr.net_f.l0[ide]

                lc = norm(tr.net_c.pts[v,:] .- tr.net_c.pts[u,:])
                l0c = tr.net_c.l0[ide]
                Gradient[ide] +=  (lc - l0c)^2 - (lf - l0f)^2    
            end
        end
        #update current free strains
        for (ido,(edge,_,_,l0)) in enumerate(tr.output)
            sf_new[ido] += cal_strain(tr.net_f, edge, l0)
        end
    end

    Gradient=Gradient ./ step_md
    sf_new .= sf_new ./ step_md
    
    # stiffness update
    learn_k!(tr, Gradient, alpha)
   
    return sf_new # return current free strains on output edges 
end

