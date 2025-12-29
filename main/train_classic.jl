using PhyLearn_EN
dim=parse(Int, ARGS[1])
network_id=parse(Int, ARGS[2])
trainT=parse(Float64, ARGS[3])

alpha=1.0
trainsteps=2_000_001
save_interval=200_000
print_interval=40_000
taskid=1

println("Dimension: $(dim), Network ID: $(network_id), taskid: $(taskid), Training Temperature: $(trainT)")

root="/data2/shared/yueshang/julia/classic/"
net_file = joinpath(root, "/dim$(dim)/network$(network_id)/network.txt")
task_path = joinpath(root, "/dim$(dim)/network$(network_id)/task$(taskid)/")
net=ENM(net_file)
input,output=load_task(task_path)
train0=Trainer_CL(net,input,output)

#classical eta=1 training boundary condition
for op in train0.input
        set_edge_k!(train0.net_f,op[1],100.0)
        set_edge_l0!(train0.net_f,op[1],(op[2]+1)*op[3])
        set_edge_k!(train0.net_c,op[1],100.0)
        set_edge_l0!(train0.net_c,op[1],(op[2]+1)*op[3])'
end
for op in train0.output
        set_edge_k!(train0.net_c,op[1],100.0)
        set_edge_l0!(train0.net_c,op[1],(op[2]+1)*op[3])
end



# training loop
trainpath=joinpath(task_path, "trainT$(trainT)_alpha$(alpha)_classic/")

t0=time()
for stepid in 1:trainsteps
    Gradient=zeros(length(train0.net_f.edges))
    run_md!(train0.net_f,trainT)
    run_md!(train0.net_c,trainT)
    update_grad!(train0, Gradient)
    update_k!(train0, Gradient, alpha)
    if stepid%print_interval==0
        E_f=cal_elastic_energy(train0.net_f)
        E_c=cal_elastic_energy(train0.net_c)
        strain_f_out=cal_strain(train0.net_f,train0.output[1][1])# this is only for one output edge
        println("Step: $(stepid), E_f: $(E_f), E_c: $(E_c), strain: $(strain_f_out)")
    end

    if stepid%save_interval==0
        save_k(train0.net_f, joinpath(trainpath, "k$(stepid).f64"))
        save_pts(train0.net_f, joinpath(trainpath, "pts$(stepid).f64"))
        
    end
    
end
println("trainingtime = ", time() - t0, " s")

