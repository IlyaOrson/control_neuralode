# adapted from https://diffeqflux.sciml.ai/dev/examples/feedback_control/

module ControlNeuralODE

using Dates
using Base.Filesystem

using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, Zygote, GalacticOptim, DiffEqSensitivity
using UnicodePlots: lineplot, lineplot!, histogram
using CSV, Tables

function fun_plotter(fun, array; xlim=(0,0))
    output = map(fun, eachrow(array)...)
    lineplot(
        output;
        title="Custom Function",
        name="fun",
        ylim=extrema(output),
        xlim
    )
end

# handy terminal plots
function unicode_plotter(states, controls; only=nothing, vars=nothing, fun=nothing)
    @assert size(states, 2) == size(controls, 2)
    xlim = (0, size(states, 2))
    if only == :states
        if !isnothing(fun)
            return fun_plotter(fun, states; xlim)
        end
        typeof(vars) <: Vector && (states = @view states[vars, :])
        ylim = states |> extrema
        tag = isnothing(vars) ? "x1" : "x$(vars[1])"
        plt = lineplot(
            states[1,:];
            title = "State Evolution",
            name = tag,
            xlabel = "step",
            ylim, xlim
        )
        for (i, s) in enumerate(eachrow(states[2:end,:]))
            tag = isnothing(vars) ? "x$(i+1)" : "x$(vars[i+1])"
            lineplot!(plt, collect(s), name = tag)
        end
    elseif only == :controls
        if !isnothing(fun)
            return fun_plotter(fun, controls; xlim)
        end
        typeof(vars) <: Vector && (controls = @view controls[vars, :])
        ylim = controls |> extrema
        tag = isnothing(vars) ? "c1" : "c$(vars[1])"
        plt = lineplot(
            controls[1,:];
            title = "Control Evolution",
            name = tag,
            xlabel = "step",
            ylim, xlim
        )
        for (i, s) in enumerate(eachrow(controls[2:end,:]))
            tag = isnothing(vars) ? "c$(i+1)" : "c$(vars[i+1])"
            lineplot!(plt, collect(s), name = tag)
        end
    else
        if !isnothing(fun)
            return fun_plotter(fun, hcat(states, controls); xlim)
        end
        ylim = Iterators.flatten((states, controls)) |> extrema
        plt = lineplot(
            states[1,:];
            title = "State and Control Evolution",
            name = "x1",
            xlabel = "step",
            ylim, xlim
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


function generate_data(prob, params, tsteps)

    # integrate with given parameters
    solution = solve(prob, AutoTsit5(Rosenbrock23()), p = params, saveat = tsteps)

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
    return solution.t, states, controls
end

string_datetime() = replace(string(now()), (":" => "_"))

function store_simulation(name, prob, params, tsteps; metadata=nothing, current_datetime=nothing, filename=nothing)

    times, states, controls = generate_data(prob, params, tsteps)

    parent = dirname(@__DIR__)
    isnothing(current_datetime) && (current_datetime = string_datetime())
    datadir = joinpath(
        parent, "data",
        basename(name),
        current_datetime
    )
    @info "Storing data in $datadir"
    mkpath(datadir)

    # time_data = Tables.table(reshape(solution.t, :, 1), header = ("t"))
    # CSV.write(joinpath(datadir, "time.csv"), time_data)

    state_headers = ["x$i" for i in 1:size(states, 1)]
    # state_data = Tables.table(states', header = state_headers)
    # CSV.write(joinpath(datadir, "states.csv"), state_data)

    control_headers = ["c$i" for i in 1:size(controls, 1)]
    # control_data = Tables.table(controls', header = control_headers)
    # CSV.write(joinpath(datadir, "controls.csv"), control_data)

    full_data = Tables.table(
        hcat(times, states', controls'),
        header = vcat(["t"], state_headers, control_headers)
    )
    isnothing(filename) ? filename="data.csv" : filename=filename*".csv"
    CSV.write(joinpath(datadir, filename), full_data)

    !isnothing(metadata) && CSV.write(joinpath(datadir, "metadata.txt"), metadata; writeheader=false, delim="\t")
end


# simulate evolution at each iteration and plot it
function plot_simulation(
    prob, params, tsteps;
    show=nothing, only=nothing, vars=nothing, fun=nothing, yrefs=nothing
)

    !isnothing(show) && @show show

    times, states, controls = generate_data(prob, params, tsteps)
    plt = unicode_plotter(states, controls; only, vars, fun)
    if !isnothing(yrefs)
        for yref in yrefs
            lineplot!(plt, x -> yref, name="$yref")
        end
    end
    display(plt)
    return false  # if return true, then optimization stops
end

function controller_shape(controller)
    # this method is brittle as any function inside the Chain
    # will not be identified, could be a problem if those change dimensions

    # Flux Layers have fields (:weight, :bias, :σ)
    # FastLayers have fields (:out, :in, :σ, :initial_params, :bias)
    dims_input = [l.in for l in controller.layers[1:end] if typeof(l) <: FastDense]
    dims_output = [l.out for l in controller.layers[1:end] if typeof(l) <: FastDense]
    push!(dims_input, pop!(dims_output))
end


function runner(script)
    include(joinpath(@__DIR__, endswith(script, ".jl") ? script : "$script.jl"))
end


end # module
