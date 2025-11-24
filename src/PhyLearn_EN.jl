module PhyLearn_EN

include("ENM.jl")
include("TrainerCL.jl")

using .ENM
using .TrainerCL

export ENM, Trainer_CL, step!, run_md!, calc_strain_f, clamp_eta!, learn_k!, calc_modes

end
