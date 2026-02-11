using PhyLearn_EN
using StableRNGs

# Parse command line arguments
if length(ARGS) < 11
    error("Not enough command line arguments provided.")
end

dim = parse(Int, ARGS[1])
network_id = parse(Int, ARGS[2])
taskid = parse(Int, ARGS[3])
trainT = parse(Float64, ARGS[4])
seed = parse(Int, ARGS[5])
testT = parse(Float64, ARGS[6])
strain_source = parse(Float64, ARGS[7])
seed2 = parse(Int, ARGS[8])
traintime = parse(Int, ARGS[9])
alpha = parse(Float64, ARGS[10])

# Constants
const TIMEWINDOW = 200
const CAL_PER = 2000
const N_FRAMES = 5000
const ROOT = "/data2/shared/yueshang/julia/"

# Setup paths
net_file = joinpath(ROOT, "dim$(dim)", "network$(network_id)", "network.txt")
task_path = joinpath(ROOT, "dim$(dim)", "network$(network_id)", "task$(taskid)")
trainpath = joinpath(task_path, "trainT$(trainT)_alpha$(alpha)_tw$(TIMEWINDOW)", "seed$(seed)")
data_path = joinpath(trainpath, "Susceptibility_testT$(testT)_time$(traintime)_strain$(strain_source)_seed$(seed2)")

# Load data
enm = ENM(net_file)
input, output = load_task(task_path)
load_k(enm, joinpath(trainpath, "k$(traintime).f64"))


# Apply strain if specified
if strain_source != 0.0
    put_strain!(enm, input[1][1], strain_source)
end

mkpath(data_path)

function cal_dl2(enm::ENM)
    """Calculate (li - li0)^2 for all edges."""
    pts = enm.pts
    edges = enm.edges
    dl2 = Vector{Float64}(undef, size(edges, 1))
    
    @inbounds for i in axes(edges, 1)
        idx1, idx2 = edges[i, 1], edges[i, 2]
        delta = pts[idx1, :] .- pts[idx2, :]
        dl2[i] = (sqrt(sum(delta .^ 2)) - enm.l0[i]) ^ 2
    end
    
    return dl2
end

# Main loop to calculate susceptibility
ne=size(enm.edges, 1)
OiOj=Matrix{Float64}(undef,ne,ne)
Oi=Vector{Float64}(undef,ne)
for stepid in 1:N_FRAMES
    rng = StableRNG(seed2 + stepid)
    run_md!(enm, testT, steps=CAL_PER, rng=rng)
    dl2 = cal_dl2(enm)
    Oi .+= dl2
    OiOj .+= dl2 * dl2'
end

Oi ./= N_FRAMES
OiOj ./= N_FRAMES
susceptibility = OiOj .- Oi * Oi'

open(joinpath(data_path, "susceptibility.f32"), "w") do file
    write(file, Float32.(susceptibility))
end