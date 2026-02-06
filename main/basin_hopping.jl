# basin_hopping to determine the inherent states
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

record_per = 200000 #1000tau
n_frames= 1000 #number of basin hopping steps

root="/data2/shared/yueshang/julia/"
net_file =  "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/network.txt"
task_path = "/data2/shared/yueshang/julia/dim$(dim)/network$(network_id)/task$(taskid)/"

enm=ENM(net_file)
input,output=load_task(task_path)
trainpath=joinpath(task_path, "trainT$(trainT)_alpha$(alpha)_tw$(timewindow)","seed$(seed)")
load_k(enm, joinpath(trainpath, "k$(traintime).f64"))

#put strain
if strain_source != 0.0
     put_strain!(enm, input[1][1], strain_source)
end

inherent_path=joinpath(trainpath, "IS_testT$(testT)_time$(traintime)_strain$(strain_source)_seed$(seed2)/")
mkpath(inherent_path)
@inline vec_pts(pts::Matrix{Float64}) = reshape(permutedims(pts), :, 1)[:,1]
for stepid in 1:n_frames
    
    rng=StableRNG(seed2+stepid)
    run_md!(enm,testT,steps=record_per, rng=rng)
    quench_fire!(enm)
    X = copy(enm.pts)
    rigid_align!(X, enm.pts0)
    if maximum(abs.(X .- enm.pts0)) > 1e-3:
        open(joinpath(inherent_path, "IS_step$(stepid).f32"), "w") do file
            write(file, Float32.(vec_pts(X)))
        end
    end
end
# there is a huge concern that the bounded global minima is unknow, it is better to use a hash map to distinguish all the minima that found
 