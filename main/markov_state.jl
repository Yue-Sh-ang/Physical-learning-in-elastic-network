# To record locations of the markov state model
using PhyLearn_EN
using StableRNGs

dim=parse(Int, ARGS[1])
network_id=parse(Int, ARGS[2])
taskid=parse(Int, ARGS[3])
trainT=parse(Float64, ARGS[4])
seed=parse(Int, ARGS[5])
testT=parse(Float64, ARGS[6])


function simulate_and_record(
    enm::ENM;
    steps::Int,
    save_every::Int,
    Φ::Matrix{Float64},
    out_edge::Int,
    advance!::Function,
    savepath::Union{Nothing,String}=nothing,
 )
    k = size(Φ, 2)
    nframes = Int(floor(steps / save_every)) + 1

    A    = Matrix{Float64}(undef, nframes, k)
    sout = Vector{Float64}(undef, nframes)
    t    = Vector{Int}(undef, nframes)
    frame = 1
    A[frame, :] .= project_modes_rigid(enm, Φ)   # <<< UPDATED LINE
    sout[frame] = cal_strain(enm, out_edge)
    t[frame] = 0

    # ---- MD loop ----
    for s in 1:steps
        advance!(enm)

        if s % save_every == 0
            frame += 1
            A[frame, :] .= project_modes_rigid(enm, Φ)  # <<< UPDATED LINE
            sout[frame] = cal_strain(enm, out_edge)
            t[frame] = s
        end
    end

    if savepath !== nothing
        @save savepath A sout t
    end

    return (A=A, sout=sout, t=t)
end

traintime=10_000
alpha=50.0
timewindow=200

println("Dimension: $(dim), Network ID: $(network_id), taskid: $(taskid) trainT : $(trainT) testT: $(testT)")

root="/data2/shared/yueshang/julia/"
net_file =  "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/network.txt"
task_path = "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/task$(taskid)/"

enm=ENM(net_file)
input,output=load_task(task_path)

load_k(enm, joinpath(task_path, "trainT$(trainT)_alpha$(alpha)_tw$(timewindow)/seed$(seed)/k$(traintime).f64"))
dim=enm.dim
if dim==2
    m0_num=3
elseif dim==3
    m0_num=6

Φ, λ = compute_soft_modes(enm, 10+m0_num)

