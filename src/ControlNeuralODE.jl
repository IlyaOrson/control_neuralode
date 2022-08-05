module ControlNeuralODE

using Base: @kwdef, ifelse
using Base.Filesystem: mkpath
using DelimitedFiles: readdlm, writedlm

using Dates: now
using DataStructures: SortedDict
using GarishPrint: pprint
using ArgCheck: @argcheck, @check
using Formatting: format, sprintf1
using Suppressor: @capture_out
using ProgressMeter: ProgressMeter, Progress, ProgressUnknown
# using ProgressLogging: @withprogress, @logprogress
using Infiltrator: @infiltrate
using Statistics: mean, std
using LineSearches: BackTracking
using InfiniteOpt:
    InfiniteOpt,
    Infinite,
    InfiniteModel,
    OrthogonalCollocation,
    @infinite_parameter,
    @variable,
    @variables,
    @constraint,
    @constraints,
    @objective,
    integral,
    set_start_value_function,
    optimizer_with_attributes,
    optimizer_model,
    solution_summary,
    raw_status,
    termination_status,
    has_values,
    objective_value,
    supports,
    value,
    ∂
using Ipopt: Ipopt
using LBFGSB: LBFGSB
using Optim: Optim, LBFGS, BFGS
using DataInterpolations: LinearInterpolation
using ApproxFun: Chebyshev, Fun, Interval  # https://github.com/stevengj/FastChebInterp.jl
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote, pullback
using Flux: Flux, glorot_uniform, sigmoid_fast, tanh_fast, ADAMW
using SciMLBase:
    ODEProblem,
    DECallback,
    remake,
    AbstractODEAlgorithm,
    AbstractSensitivityAlgorithm,
    AbstractODEProblem
using DiffEqCallbacks: FunctionCallingCallback
using OrdinaryDiffEq: solve, AutoTsit5, Rodas4P, Rosenbrock23, Tsit5, QNDF, FBDF
using SciMLSensitivity:
    # discrete forward
    ForwardDiffSensitivity,
    # discrete adjoint
    ReverseDiffAdjoint,
    TrackerAdjoint,
    ZygoteAdjoint,
    # continuous forward
    ForwardSensitivity,
    # continuous adjoint
    InterpolatingAdjoint,
    QuadratureAdjoint,
    BacksolveAdjoint,
    TrackerVJP,
    ZygoteVJP,
    EnzymeVJP,
    ReverseDiffVJP
using DiffEqFlux: FastChain, FastDense, initial_params, FastLayer  # sciml_train
using UnicodePlots: lineplot, lineplot!, histogram
using Serialization: serialize, deserialize
using JSON3: JSON3

using LazyGrids: ndgrid
using PyPlot: plt, matplotlib, ColorMap, plot3D, scatter3D
using LaTeXStrings: @L_str

import CommonSolve: solve

# scripts as functions
export batch_reactor, bioreactor, semibatch_reactor, van_der_pol

# NOTE : mark the variables that work as constants (avoid constants for Revise.jl)
@show const INTEGRATOR = Tsit5()

# https://diffeq.sciml.ai/stable/analysis/sensitivity/#Choosing-a-Sensitivity-Algorithm

# Discrete sensitivity analysis
# SENSEALG = ForwardSensitivity()
# SENSEALG = ReverseDiffAdjoint()
# SENSEALG = TrackerAdjoint()
# SENSEALG = ZygoteAdjoint()

# Continuous sensitivity analysis
# SENSEALG = ForwardDiffSensitivity()
const SENSEALG = QuadratureAdjoint(; autojacvec=ReverseDiffVJP())
# SENSEALG = QuadratureAdjoint(; autojacvec=ZygoteVJP())
# SENSEALG = QuadratureAdjoint(; autojacvec=TrackerVJP())
# SENSEALG = QuadratureAdjoint(; autojacvec=EnzymeVJP())
# SENSEALG = InterpolatingAdjoint(; autojacvec=ReverseDiffVJP(), checkpointing=true)
# SENSEALG = InterpolatingAdjoint(; autojacvec=ZygoteVJP(), checkpointing=true)
# SENSEALG = InterpolatingAdjoint(; autojacvec=TrackerVJP(), checkpointing=true)
# SENSEALG = InterpolatingAdjoint(; autojacvec=EnzymeVJP(), checkpointing=true)
@show SENSEALG

include("auxiliaries.jl")
include("controlODE.jl")
include("nn.jl")
include("penalties.jl")
include("interpolation.jl")
include("simulators.jl")
include("training.jl")
include("plotting.jl")
include("loading.jl")

# case studies
include("systems/batch_reactor.jl")
include("systems/van_der_pol.jl")
include("systems/bioreactor.jl")
include("systems/semibatch_reactor.jl")

# classic collocation
include("collocation/classic/van_der_pol.jl")
include("collocation/classic/bioreactor.jl")
include("collocation/classic/semibatch_reactor.jl")

# training
include("scripts/batch_reactor.jl")
include("scripts/van_der_pol.jl")
include("scripts/bioreactor.jl")
include("scripts/semibatch_reactor.jl")

end # module
