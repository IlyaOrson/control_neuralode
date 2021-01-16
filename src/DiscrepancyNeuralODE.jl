# adapted from https://diffeqflux.sciml.ai/dev/examples/feedback_control/

module DiscrepancyNeuralODE

using Dates
using Base.Filesystem

using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, Zygote, GalacticOptim, DiffEqSensitivity
using UnicodePlots: lineplot, lineplot!
using ClearStacktrace  # nicer stacktraces (unnecesary in julia 1.6)
using CSV, Tables

# handy terminal plots
function unicode_plotter(states, controls; only=nothing, vars=nothing)
    if only == :states
        typeof(vars) <: Vector && (states = @view states[vars, :])
        ylim = states |> extrema
        tag = isnothing(vars) ? "x1" : "x$(vars[1])"
        plt = lineplot(
            states[1,:],
            title = "State Evolution",
            name = tag,
            xlabel = "step",
            ylim = ylim,
        )
        for (i, s) in enumerate(eachrow(states[2:end,:]))
            tag = isnothing(vars) ? "x$(i+1)" : "x$(vars[i+1])"
            lineplot!(plt, collect(s), name = tag)
        end
    elseif only == :controls
        typeof(vars) <: Vector && (controls = @view controls[vars, :])
        ylim = controls |> extrema
        tag = isnothing(vars) ? "c1" : "c$(vars[1])"
        plt = lineplot(
            controls[1,:],
            title = "Control Evolution",
            name = tag,
            xlabel = "step",
            ylim = ylim,
        )
        for (i, s) in enumerate(eachrow(controls[2:end,:]))
            tag = isnothing(vars) ? "c$(i+1)" : "c$(vars[i+1])"
            lineplot!(plt, collect(s), name = tag)
        end
    else
        ylim = Iterators.flatten((states, controls)) |> extrema
        plt = lineplot(
            states[1,:],
            title = "State and Control Evolution",
            name = "x1",
            xlabel = "step",
            ylim = ylim,
        )
        for (i, s) in enumerate(eachrow(states[2:end,:]))
            lineplot!(plt, collect(s), name = "x$(i+1)")
        end
        for (i, c) in enumerate(eachrow(controls))
            lineplot!(plt, collect(c), name = "c$i")
        end
    end
    return plt
end

# simulate evolution at each iteration and plot it
function plot_simulation(params, loss, prob, tsteps; only=nothing, vars=nothing, store=nothing)
    @info "Objective" loss
    solution = solve(prob, Tsit5(), p = params, saveat = tsteps)

    # construct arrays with the same type used by the integrator
    elements_type = eltype(solution.t)
    states = Array(solution)
    total_steps = size(states, 2)
    state_dimension = size(states, 1)
    control_dimension = length(controller(solution.u[1], params))

    # regenerate controls from controller
    controls = zeros(elements_type, control_dimension, total_steps)
    for (step, state) in enumerate(solution.u)
        controls[:, step] = controller(state, params)
    end
    if typeof(store) == String
        parent = dirname(@__DIR__)
        current_datetime = replace(string(now()), (":" => "_"))
        datadir = joinpath(
            parent, "data",
            basename(store),
            current_datetime
        )
        @info "Storing data in $datadir"
        @show Base.Filesystem.ispath(joinpath(datadir, "states.csv"))
        mkpath(datadir)

        # time_data = Tables.table(reshape(solution.t, :, 1), header = ("t"))
        # CSV.write(joinpath(datadir, "time.csv"), time_data)

        state_headers = ["x$i" for i in 1:state_dimension]
        # state_data = Tables.table(states', header = state_headers)
        # CSV.write(joinpath(datadir, "states.csv"), state_data)

        control_headers = ["c$i" for i in 1:control_dimension]
        # control_data = Tables.table(controls', header = control_headers)
        # CSV.write(joinpath(datadir, "controls.csv"), control_data)

        full_data = Tables.table(
            hcat(solution.t, states', controls'),
            header = vcat(["t"], state_headers, control_headers)
        )
        CSV.write(joinpath(datadir, "data.csv"), full_data)
    end

    display(unicode_plotter(states, controls; only, vars))
    return false  # if return true, then optimization stops
end

function runner(script)
    include(joinpath(@__DIR__, "$script.jl"))
end
# runner("van_der_pol")  # for PackageCompiler
end # module
