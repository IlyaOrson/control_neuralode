module ControlNeuralODE

using Base: @kwdef, ifelse
using Base.Filesystem: mkpath

using Dates: now
using LazyGrids: ndgrid
using ArgCheck: @argcheck
using Formatting: format, sprintf1
using ProgressMeter: Progress, next!
using Infiltrator: @infiltrate
using Statistics: mean
using LineSearches: BackTracking
using InfiniteOpt:
    Infinite,
    InfiniteModel,
    OrthogonalCollocation,
    @infinite_parameter,
    @variables,
    @constraint,
    @constraints,
    @objective,
    set_start_value_function,
    optimizer_with_attributes,
    optimizer_model,
    optimize!,
    solution_summary,
    supports,
    value,
    ∂
using Ipopt: Ipopt
using Optim: LBFGS, BFGS
using ApproxFun: Chebyshev, Fun, (..)
using Zygote: Zygote
using Flux: glorot_uniform, ADAM, NADAM, sigmoid_fast, tanh_fast
using SciMLBase: ODEProblem, DECallback, remake
using DiffEqCallbacks: FunctionCallingCallback
using OrdinaryDiffEq: Tsit5, solve #, AutoTsit5, Rosenbrock23
using DiffEqFlux: FastChain, FastDense, initial_params, sciml_train
using DiffEqSensitivity: InterpolatingAdjoint, QuadratureAdjoint, ZygoteVJP, ReverseDiffVJP
using UnicodePlots: lineplot, lineplot!
using BSON: BSON
using JSON3: JSON3
using CSV: CSV
using Tables: table

# using PyCall: PyObject
using PyPlot: plt, matplotlib, ColorMap

export batch_reactor, van_der_pol, reference_tracking, bioreactor, semibatch_reactor

include("auxiliaries.jl")
include("plotting.jl")
include("simulators.jl")
include("training.jl")

# case studies
include("systems/batch_reactor.jl")
include("systems/van_der_pol.jl")
include("systems/reference_tracking.jl")
include("systems/bioreactor.jl")
include("systems/semibatch_reactor.jl")

end # module
