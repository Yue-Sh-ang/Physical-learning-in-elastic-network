# to build the markov state model, recording trajectory data during MD,
# documenting the projections onto the softest modes
# then do TICA and clustering to build MSM

using LinearAlgebra
using Random
using Statistics
using JLD2
using Clustering

function rigid_align!(X::Matrix{Float64}, Xref::Matrix{Float64})
    # subtract centroids
    cX    = mean(X; dims=1)
    cXref = mean(Xref; dims=1)

    Xc    = X    .- cX
    Xrefc = Xref .- cXref

    # covariance
    H = Xc' * Xrefc
    U, _, Vt = svd(H)
    R = Vt' * U'

    # reflection fix
    if det(R) < 0
        Vt[end, :] .*= -1
        R = Vt' * U'
    end

    # apply rotation + translation
    X .= (Xc * R) .+ cXref
    return nothing
end

@inline vec_pts(pts::Matrix{Float64}) = reshape(pts, :, 1)[:,1]

function mode_basis(enm::ENM; k::Int=10)
    #dim=2 or 3 only
    if enm.dim == 2
        skip = 3   # 2 translations + 1 rotation
    else
        skip = 6   # 3 translations + 3 rotations
    end

    # Jacobian evaluated at pts0 by construction
    E = cal_modes(enm, false)
    p = sortperm(E.values)          # ascending
    p = p[(skip+1):min(skip+k, end)]

    Φ = E.vectors[:, p]             # ndof × k
    λ = E.values[p]
    return Φ, λ
end

"""
    project_modes_rigid(enm, Φ)

Rigid-align current configuration to pts0, then project onto soft modes.
"""
function project_modes_rigid(enm::ENM, Φ::Matrix{Float64})
    # make a working copy (do NOT modify enm.pts permanently)
    X = copy(enm.pts)

    # rigid alignment
    rigid_align!(X, enm.pts0)

    # displacement vector
    x  = vec_pts(X)
    x0 = vec_pts(enm.pts0)

    return Φ' * (x .- x0)
end


#usage can be find at chatgpt... take a break now :)