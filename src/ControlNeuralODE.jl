module ControlNeuralODE

using Dates
using Base.Filesystem

using ProgressMeter
using Statistics: mean
using LineSearches, Optim, GalacticOptim
using Zygote, Flux
using OrdinaryDiffEq, DiffEqSensitivity, DiffEqFlux
using UnicodePlots: lineplot, lineplot!, histogram, boxplot
using BSON, JSON3, CSV, Tables

export batch_reactor, van_der_pol, reference_tracking, bioreactor, semibatch_reactor

function fun_plotter(fun, array; xlim=(0, 0))
    output = map(fun, eachrow(array)...)
    return lineplot(output; title="Custom Function", name="fun", ylim=extrema(output), xlim)
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
            states[1, :]; title="State Evolution", name=tag, xlabel="step", ylim, xlim
        )
        for (i, s) in enumerate(eachrow(states[2:end, :]))
            tag = isnothing(vars) ? "x$(i+1)" : "x$(vars[i+1])"
            lineplot!(plt, collect(s); name=tag)
        end
    elseif only == :controls
        if !isnothing(fun)
            return fun_plotter(fun, controls; xlim)
        end
        typeof(vars) <: Vector && (controls = @view controls[vars, :])
        ylim = controls |> extrema
        tag = isnothing(vars) ? "c1" : "c$(vars[1])"
        plt = lineplot(
            controls[1, :]; title="Control Evolution", name=tag, xlabel="step", ylim, xlim
        )
        for (i, s) in enumerate(eachrow(controls[2:end, :]))
            tag = isnothing(vars) ? "c$(i+1)" : "c$(vars[i+1])"
            lineplot!(plt, collect(s); name=tag)
        end
    else
        if !isnothing(fun)
            return fun_plotter(fun, hcat(states, controls); xlim)
        end
        ylim = Iterators.flatten((states, controls)) |> extrema
        plt = lineplot(
            states[1, :];
            title="State and Control Evolution",
            name="x1",
            xlabel="step",
            ylim,
            xlim,
        )
        for (i, s) in enumerate(eachrow(states[2:end, :]))
            lineplot!(plt, collect(s); name="x$(i+1)")
        end
        for (i, c) in enumerate(eachrow(controls))
            lineplot!(plt, collect(c); name="c$i")
        end
    end
    return plt
end

function generate_data(controller, prob, params, tsteps)

    # integrate with given parameters
    solution = solve(prob, AutoTsit5(Rosenbrock23()); p=params, saveat=tsteps)

    # construct arrays with the same type used by the integrator
    elements_type = eltype(solution.t)
    states = Array(solution)
    total_steps = size(states, 2)
    # state_dimension = size(states, 1)
    control_dimension = length(controller(solution.u[1], params))

    # regenerate controls from controller
    controls = zeros(elements_type, control_dimension, total_steps)
    for (step, state) in enumerate(solution.u)
        controls[:, step] = controller(state, params)
    end
    return solution.t, states, controls
end

string_datetime() = replace(string(now()), (":" => "_"))

function generate_data_subdir(callerfile; current_datetime=nothing)
    parent = dirname(@__DIR__)
    isnothing(current_datetime) && (current_datetime = string_datetime())
    datadir = joinpath(parent, "data", basename(callerfile), current_datetime)
    @info "Generating data directory: $datadir"
    mkpath(datadir)
    return datadir
end

function store_simulation(
    filename, datadir, controller, prob, params, tsteps; metadata=nothing
)
    controller_file = joinpath(datadir, filename * ".bson")
    BSON.@save controller_file controller

    times, states, controls = generate_data(controller, prob, params, tsteps)

    state_headers = ["x$i" for i in 1:size(states, 1)]
    control_headers = ["c$i" for i in 1:size(controls, 1)]

    full_data = Tables.table(
        hcat(times, states', controls'); header=vcat(["t"], state_headers, control_headers)
    )

    CSV.write(joinpath(datadir, filename * ".csv"), full_data)

    if !isnothing(metadata)
        open(joinpath(datadir, "metadata.json"), "w") do f
            JSON3.pretty(f, JSON3.write(metadata))
            println(f)
        end
    end
end

# simulate evolution at each iteration and plot it
function plot_simulation(
    controller,
    prob,
    params,
    tsteps;
    show=nothing,
    only=nothing,
    vars=nothing,
    fun=nothing,
    yrefs=nothing,
)
    !isnothing(show) && @show show

    # TODO: use times in plotting?
    times, states, controls = generate_data(controller, prob, params, tsteps)
    plt = unicode_plotter(states, controls; only, vars, fun)
    if !isnothing(yrefs)
        for yref in yrefs
            lineplot!(plt, x -> yref; name="$yref")
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
    return push!(dims_input, pop!(dims_output))
end

# Feller, C., & Ebenbauer, C. (2014).
# Continuous-time linear MPC algorithms based on relaxed logarithmic barrier functions.
# IFAC Proceedings Volumes, 47(3), 2481–2488.
# https://doi.org/10.3182/20140824-6-ZA-1003.01022

function quadratic_relaxation(z, δ)
    return one(typeof(z)) / 2 * (((z - 2δ) / δ)^2 - one(typeof(z))) - log(δ)
end
exponential_relaxation(z, δ) = exp(one(typeof(z)) - z / δ) - one(typeof(z)) - log(δ)
function relaxed_log_barrier(z; δ=0.3f0)
    return max(z > δ ? -log(z) : exponential_relaxation(z, δ), zero(typeof(z)))
end
function relaxed_log_barrier(z, lower, upper; δ=(upper - lower) / convert(typeof(z), 2))
    return relaxed_log_barrier(z - lower; δ) + relaxed_log_barrier(upper - z; δ)
end

indicator_function(z; δ) = min(z, zero(typeof(z)))

function preconditioner(
    controller,
    precondition,
    system!,
    t0,
    u0,
    time_fractions;
    reg_coeff=1f-1,
    f_tol=1f-3,
    saveat=(),
    progressbar=true,
    control_range_scaling=nothing,
)
    θ = initial_params(controller)
    fixed_dudt!(du, u, p, t) = system!(du, u, p, t, precondition, :time)
    prog = Progress(
        length(time_fractions);
        desc="Pretraining in subintervals...",
        dt=0.5,
        showspeed=true,
        enabled=progressbar,
    )
    for partial_time in time_fractions
        tspan = (t0, partial_time)
        fixed_prob = ODEProblem(fixed_dudt!, u0, tspan)
        fixed_sol = solve(fixed_prob, BS3(); abstol=1f-1, reltol=1f-1, saveat)

        function precondition_loss(params; plot=false)

            plot_arrays = Dict(
                :reference => [],
                :control => [],
            )
            sum_squares = 0.0f0

            # for (time, state) in zip(fixed_sol.t, fixed_sol.u)  # Zygote error
            for (i, state) in enumerate(eachcol(Array(fixed_sol)))
                reference = precondition(fixed_sol.t[i], nothing)  # precondition(time, params)
                control = controller(state, params)
                diff_square = (control - reference) .^ 2
                if !isnothing(control_range_scaling)
                    diff_square ./ control_range_scaling
                end
                sum_squares += sum(diff_square)
                Zygote.ignore() do
                    if plot
                        push!(plot_arrays[:reference], reference)
                        push!(plot_arrays[:control], control)
                    end
                end
            end
            Zygote.ignore() do
                if plot
                    refereces = reduce(hcat, plot_arrays[:reference])
                    controls = reduce(hcat, plot_arrays[:control])
                    @assert length(refereces) == length(controls)
                    for r in 1:size(refereces, 1)
                        p = lineplot(refereces[r,:]; name="fixed")
                        lineplot!(p, controls[r,:]; name="neural")
                        display(p)
                    end
                end
            end
            regularization = reg_coeff * mean(abs2, params)
            return sum_squares + regularization
        end

        @time preconditioner = DiffEqFlux.sciml_train(
            precondition_loss,
            θ,
            BFGS(; initial_stepnorm=0.01);
            # maxiters=10,
            allow_f_increases=true,
            f_tol,
        )
        θ = preconditioner.minimizer
        @show ploss = precondition_loss(θ; plot=true)
        next!(prog; showvalues=[(:loss, ploss)])
    end
    return θ
end

function constrained_training(
    controller,
    prob,
    θ,
    loss;
    αs,
    δs,
    tsteps=(),
    datadir=nothing,
    show_progresbar=false,
    plots_callback=nothing,
    f_tol=1f-1,
    metadata=Dict(),
)
    @assert length(αs) == length(δs)
    prog = Progress(
        length(αs); desc="Training with constraints...", enabled=show_progresbar
    )
    for (α, δ) in zip(αs, δs)

        # prob = ODEProblem(dudt!, u0, tspan, θ)
        prob = remake(prob; p=θ)

        # closure to comply with optimization interface
        loss_(params) = reduce(+, loss(params, prob; α, δ, tsteps))

        if !isnothing(plots_callback)
            plots_callback(controller, prob, θ, tsteps)
        end

        # function print_callback(params, loss)
        #     println(loss)
        #     return false
        # end

        @time result = DiffEqFlux.sciml_train(
            loss_,
            θ,
            LBFGS(; linesearch=LineSearches.BackTracking());
            # cb=print_callback,
            allow_f_increases=true,
            f_tol,
        )
        θ = result.minimizer

        objective, state_penalty, control_penalty, regularization = loss(
            θ, prob; α, δ, tsteps
        )

        @info "Current values" α, δ, objective, state_penalty, control_penalty, regularization

        if !isnothing(datadir)
            local_metadata = Dict(
                :objective => objective,
                :state_penalty => state_penalty,
                :control_penalty => control_penalty,
                :regularization_cost => regularization,
                :parameters => θ,
                :num_params => length(initial_params(controller)),
                :layers => controller_shape(controller),
                :penalty_relaxations => δs,
                :penalty_coefficients => αs,
                :tspan => prob.tspan,
                :tsteps => tsteps,
            )
            metadata = merge(metadata, local_metadata)
            store_simulation(
                "delta_$(round(δ, digits=2))",
                datadir,
                controller,
                prob,
                θ,
                tsteps;
                metadata,
            )
        end
        # ProgressMeter.finish!(prog)
        # break
        ProgressMeter.next!(
            prog;
            showvalues=[
                (:α, α), (:δ, δ), (:objective, objective), (:state_penalty, state_penalty)
            ],
        )
    end
    return θ
end

# function runner(script)
#     return include(joinpath(@__DIR__, endswith(script, ".jl") ? script : "$script.jl"))
# end

include("batch_reactor.jl")
include("van_der_pol.jl")
include("reference_tracking.jl")
include("bioreactor.jl")
include("semibatch_reactor.jl")

end # module
