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
using Infiltrator: @infiltrate
using StaticArrays: SA
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
using DiffEqSensitivity:
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
using Random: default_rng
using ComponentArrays: ComponentArray, getaxes
using Lux: Lux, Chain, Dense, AbstractExplicitLayer, initialparameters, initialstates
using DiffEqFlux: FastChain, FastDense, initial_params, FastLayer  # sciml_train
using UnicodePlots: lineplot, lineplot!, histogram
using Serialization: serialize, deserialize
using JSON3: JSON3

using LazyGrids: ndgrid
using PyPlot: plt, matplotlib, ColorMap, plot3D, scatter3D
using LaTeXStrings: @L_str

import CommonSolve: solve

export batch_reactor, bioreactor, semibatch_reactor
export van_der_pol, van_der_pol_direct

# TODO: mark the variables that work as constants (avoid constants for Revise.jl)
@show INTEGRATOR = Tsit5()

# https://diffeqflux.sciml.ai/stable/ControllingAdjoints/#Choosing-a-sensealg-in-a-Nutshell

# Discrete sensitivity analysis
# SENSEALG = ForwardSensitivity()
# SENSEALG = ReverseDiffAdjoint()
# SENSEALG = TrackerAdjoint()
# SENSEALG = ZygoteAdjoint()

# Continuous sensitivity analysis
# SENSEALG = ForwardDiffSensitivity()
SENSEALG = QuadratureAdjoint(; autojacvec=ReverseDiffVJP())
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

# neural collocation
include("collocation/neural/van_der_pol.jl")

# training
include("scripts/batch_reactor.jl")
include("scripts/van_der_pol.jl")
include("scripts/van_der_pol_direct.jl")
include("scripts/bioreactor.jl")
include("scripts/semibatch_reactor.jl")

end # module
