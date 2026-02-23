# this code is to document the trajectory of area, which can be used to tell buckling.

# To record locations of the markov state model
# adding protential energy
using PhyLearn_EN
using StableRNGs
#train info
dim=parse(Int, ARGS[1])
network_id=parse(Int, ARGS[2])
taskid=parse(Int, ARGS[3])
trainT=parse(Float64, ARGS[4])
seed=parse(Int, ARGS[5])
#test info
testT=parse(Float64, ARGS[6])
strain_source=parse(Float64, ARGS[7])
seed2=parse(Int, ARGS[8])
traintime=parse(Int, ARGS[9])
alpha=parse(Float64, ARGS[10])

timewindow=200

record_per = 100000 #500tau
n_frames= 2000


root="/data2/shared/yueshang/julia/"
net_file =  "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/network.txt"
task_path = "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/task$(taskid)/"

enm=ENM(net_file)
input,output=load_task(task_path)
trainpath=joinpath(task_path, "trainT$(trainT)_alpha$(alpha)_tw$(timewindow)","seed$(seed)")
load_k(enm, joinpath(trainpath, "k$(traintime).f64"))
n_soft=10
phi,lambda = PhyLearn_EN.mode_basis(enm, k=n_soft)

#put strain
if strain_source != 0.0
     put_strain!(enm, input[1][1], strain_source)
end

buckling_path=joinpath(trainpath, "buckling_testT$(testT)_time$(traintime)_strain$(strain_source)_seed$(seed2)/")
mkpath(buckling_path)

function polygon_area(vertices, face)
    pts = vertices[face, :]
    x = pts[:,1]
    y = pts[:,2]

    return 0.5 * sum(x .* circshift(y, -1) .-
                     circshift(x, -1) .* y)
end

function find_faces(vertices::Matrix{Float64},
                    edges::Vector{Tuple{Int,Int}})

    # ----------------------------
    # Build adjacency dictionary
    # ----------------------------
    adj = Dict{Int, Vector{Int}}()

    for (i, j) in edges
        push!(get!(adj, i, Int[]), j)
        push!(get!(adj, j, Int[]), i)
    end

    # ----------------------------
    # Sort neighbors by angle
    # ----------------------------
    for (i, neighbors) in adj
        xi = vertices[i, 1]
        yi = vertices[i, 2]

        sort!(neighbors, by = j -> atan(
            vertices[j, 2] - yi,
            vertices[j, 1] - xi
        ))
    end

    # ----------------------------
    # Create visited directed edges
    # ----------------------------
    visited = Dict{Tuple{Int,Int}, Bool}()

    for (i, j) in edges
        visited[(i, j)] = false
        visited[(j, i)] = false
    end

    faces = Vector{Vector{Int}}()

    # ----------------------------
    # Face walking
    # ----------------------------
    for ((u0, v0), _) in visited
        if visited[(u0, v0)]
            continue
        end

        u, v = u0, v0
        start = (u, v)
        face = Int[]

        while true
            if visited[(u, v)]
                break
            end

            visited[(u, v)] = true
            push!(face, u)

            neighbors = adj[v]
            idx = findfirst(==(u), neighbors)

            # next neighbor counterclockwise
            next_idx = idx == length(neighbors) ? 1 : idx + 1
            w = neighbors[next_idx]

            u, v = v, w

            if (u, v) == start
                break
            end
        end

        if length(face) > 2
            push!(faces, face)
        end
    end
    # ----------------------------
    # Remove outer face
    # ----------------------------
    areas = [polygon_area(vertices, f) for f in faces]
    outer = argmax(abs.(areas))
    deleteat!(faces, outer)
    return faces
end


faces = find_faces(enm.pts0, enm.edges) #facelist for the original network, without the outer face
assert enm.n-length(enm.edges)+length(faces) == 1 # only one face for the 2D network
#area0=[polygon_area(enm.pts0, f) for f in faces]

#run the md and observing the area of faces to detect buckling,
#store the inherent configuration if buckling happens
areas_time = Matrix{Float32}(undef, n_frames, length(faces))

for stepid in 1:n_frames
    
    rng=StableRNG(seed2+stepid)
    run_md!(enm,testT,steps=record_per, rng=rng)
    areas = [polygon_area(enm.pts, f) for f in faces]
    areas_time[stepid, :] = Float32.(areas)
    end
end

open(joinpath(buckling_path, "areas_time.f32"), "w") do file
    write(file, areas_time)
end

open(joinpath(buckling_path, "faces.bin"), "w") do io
    write(io, Int32(length(faces)))  # number of faces
    for f in faces
        write(io, Int32(length(f)))  # face length
        write(io, Int32.(f))         # indices
    end
end 
