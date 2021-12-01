module ControlNeuralODE

using Dates
using Base: @kwdef
using Base.Filesystem
using LazyGrids: ndgrid

using ArgCheck
using ProgressMeter
using Infiltrator
using Statistics: mean
using LineSearches: BackTracking
using Optim, GalacticOptim
using InfiniteOpt, Ipopt
using ApproxFun: Chebyshev, Fun, (..)
using Zygote, Flux
using OrdinaryDiffEq, DiffEqSensitivity, DiffEqFlux
using UnicodePlots: lineplot, lineplot!, histogram, boxplot
using BSON, JSON3, CSV, Tables

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
