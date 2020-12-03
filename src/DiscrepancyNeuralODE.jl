# adapted from https://diffeqflux.sciml.ai/dev/examples/feedback_control/

module DiscrepancyNeuralODE

# using Sundials
using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, Zygote, GalacticOptim
using UnicodePlots: lineplot, lineplot!
using ClearStacktrace  # nicer stacktraces (unnecesary in julia 1.6)

# handy terminal plots
function unicode_plotter(states, controls; only=nothing, vars=nothing)
    if only == :states
        typeof(vars) <: Vector && (states = @view states[vars, :])
        ylim = states |> extrema
        plt = lineplot(
            states[1,:],
            title = "State Evolution",
            name = "x1",
            xlabel = "step",
            ylim = ylim,
        )
        for (i, s) in enumerate(eachrow(states[2:end,:]))
            lineplot!(plt, collect(s), name = "x$(i+1)")
        end
    elseif only == :controls
        typeof(vars) <: Vector && (controls = @view controls[vars, :])
        ylim = controls |> extrema
        plt = lineplot(
            controls[1,:],
            title = "Control Evolution",
            name = "c1",
            xlabel = "step",
            ylim = ylim,
        )
        for (i, s) in enumerate(eachrow(controls[2:end,:]))
            lineplot!(plt, collect(s), name = "c$(i+1)")
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
function plot_simulation(params, loss, prob, tsteps; only=nothing, vars=nothing)
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
    display(unicode_plotter(states, controls; only, vars))
    return false  # if return true, then optimization stops
end

function runner(script)
    include(joinpath(@__DIR__, "$script.jl"))
end
# runner("van_der_pol")  # for PackageCompiler
end # module
