module ControlNeuralODE

using Base: @kwdef, ifelse
using Base.Filesystem

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
    optimizer_with_attributes,
    optimizer_model,
    solution_summary,
    optimize!,
    value,
    ∂
using Ipopt
using Optim, GalacticOptim
using ApproxFun: Chebyshev, Fun, (..)
using Zygote, Flux
using SciMLBase: DECallback
using DiffEqCallbacks: FunctionCallingCallback
using OrdinaryDiffEq
using DiffEqFlux
using DiffEqSensitivity
using UnicodePlots: lineplot, lineplot!, histogram, boxplot
using BSON, JSON3, CSV, Tables

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
