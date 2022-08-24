function preconditioner(
    controlODE,
    precondition;
    θ,
    ρ=nothing,
    saveat=(),
    progressbar=false,
    plot_final=true,
    integrator=INTEGRATOR,
    max_solver_iterations=10,  # per step
    kwargs...,
)
    @info "Preconditioning..."

    prog = Progress(
        length(controlODE.tsteps[2:end]);
        desc="Pretraining in subintervals t ∈ $(controlODE.tspan)",
        dt=0.2,
        showspeed=true,
        enabled=progressbar,
    )
    # skip spurious ends from collocation
    # @withprogress name="Pretraining in subintervals t ∈ $(controlODE.tspan)" begin
    for partial_time in controlODE.tsteps[2:(end - 1)]
        partial_tspan = (controlODE.tspan[1], partial_time)

        local fixed_prob
        if controlODE.inplace
            function fixed_dudt!(du, u, p, t)
                return controlODE.system(du, u, p, t, precondition; input=:time)
            end
            fixed_prob = ODEProblem(fixed_dudt!, controlODE.u0, partial_tspan)
        else
            fixed_dudt(u, p, t) = controlODE.system(u, p, t, precondition; input=:time)
            fixed_prob = ODEProblem(fixed_dudt, controlODE.u0, partial_tspan)
        end
        fixed_sol = solve(fixed_prob, integrator; saveat)

        # Zygote ignore anything unrelated to loss function
        function precondition_loss(params; plot=nothing)
            plot_arrays = Dict(:reference => [], :control => [])

            sum_squares = 0.0f0
            datapoints = length(fixed_sol.t)
            # for (time, state) in zip(fixed_sol.t, fixed_sol.u)  # Zygote error
            for (i, state) in enumerate(eachcol(Array(fixed_sol)))
                reference = precondition(fixed_sol.t[i], nothing)  # precondition(time, params)
                control = controlODE.controller(state, params)
                diff_square = (control - reference) .^ 2
                sum_squares += sum(diff_square)
                Zygote.ignore() do
                    if !isnothing(plot)
                        push!(plot_arrays[:reference], reference)
                        push!(plot_arrays[:control], control)
                    end
                end
            end
            Zygote.ignore() do
                if !isnothing(plot)
                    @argcheck plot in [:unicode, :pyplot]
                    reference = reduce(hcat, plot_arrays[:reference])
                    control = reduce(hcat, plot_arrays[:control])

                    if plot == :unicode
                        for r in axes(reference, 1)
                            p = lineplot(reference[r, :]; name="fixed")
                            lineplot!(p, control[r, :]; name="neural")
                            display(p)
                        end
                    elseif plot == :pyplot
                        cmap = ColorMap("tab20")  # binary comparisons
                        nrows = size(reference, 1)
                        rows = 1:nrows
                        fig, axs = plt.subplots(
                            nrows,
                            1;
                            sharex="col",
                            squeeze=false,
                            constrained_layout=true,
                            #tight_layout=false,
                        )
                        for r in rows
                            ax = axs[r, 1]
                            ax.plot(
                                fixed_sol.t,
                                reference[r, :],
                                "o";
                                label="fixed",
                                color=cmap(2r - 2),
                            )
                            ax.plot(
                                fixed_sol.t,
                                control[r, :];
                                label="neural",
                                color=cmap(2r - 1),
                            )
                            ax.legend()
                        end
                        fig.supxlabel("time")
                        fig.suptitle("Preconditioning")
                        fig.show()
                    end
                end
            end
            mse = sum_squares/datapoints
            if isnothing(ρ)
                return mse
            else
                regularization = ρ * sum(abs2, params)
                return mse + regularization
            end
        end

        grad!(g, params) = g .= Zygote.gradient(precondition_loss, params)[1]
        # θ = optimize_flux(θ, precondition_loss; kwargs...)
        θ = optimize_ipopt(θ, precondition_loss, grad!; tolerance=1e-1, maxiters=max_solver_iterations, verbosity=3)

        ProgressMeter.next!(prog)
        @info "Progress $(100*partial_time/controlODE.tsteps[(end - 1)])%" partial_time final_time=controlODE.tsteps[(end - 1)]
        # @logprogress partial_time/controlODE.tsteps[(end - 1)]
        if partial_time == controlODE.tsteps[end] && plot_final
            precondition_loss(θ; plot=:pyplot)
        end
    end
    # end # @withprogress
    return θ
end

function increase_by_percentage(x, per)
    @argcheck 0 < per
    return (1f0 + 1f-2 * per) * x
end

function decrease_by_percentage(x, per)
    @argcheck 0 < per < 100
    return (1f0 - 1f-2 * per) * x
end

function evaluate_barrier(
    losses,
    controlODE,
    θ;
    α,
    δ,
    ρ,
    penalty_ratio_upper_bound=1f1,
    penalty_ratio_lower_bound=1f-1,
    verbose=false,
)
    objective, state_penalty, control_penalty, regularization = losses(controlODE, θ; α, δ, ρ)
    if isinf(state_penalty)
        return :inf
    end
    state_penalty_size = abs(state_penalty)
    other_penalties_size = abs(objective)
    # other_penalties_size = max(abs(objective), abs(control_penalty), abs(regularization))
    state_penalty_ratio = state_penalty_size / other_penalties_size
    verbose && @info "Penalty ratios" α δ objective state_penalty state_penalty_ratio
    if state_penalty_ratio > penalty_ratio_upper_bound
        return :overtight
    elseif state_penalty_ratio < penalty_ratio_lower_bound
        return :overlax
    else
        return :reasonable
    end
end

function tune_barrier(
    losses,
    controlODE,
    θ;
    α,
    ρ,
    δ,
    δ_percentage_change=10f0,
    α_percentage_change=50f0,
    max_iters=10,
    increase_alpha=true,
    verbose=false,
)
    previous_evaluation = :reasonable
    counter=0

    # @withprogress name="Tuning barrier parameters" begin
    while counter < max_iters
        counter+=1

        evaluation = evaluate_barrier(losses, controlODE, θ; α, δ, ρ, verbose)
        @info "Barrier tuning" counter evaluation
        # increase_alpha && counter == 1 && evaluation == :overtight && @infiltrate
        if evaluation == :overlax
            δ = decrease_by_percentage(δ, δ_percentage_change)

        elseif evaluation in [:overtight, :inf]
            δ = increase_by_percentage(δ, δ_percentage_change)
            # @infiltrate counter == max_iters

        elseif evaluation == :reasonable
            if previous_evaluation in [:overtight, :inf]
                if increase_alpha
                    α = increase_by_percentage(α, α_percentage_change)
                end
            else
                δ = decrease_by_percentage(δ, δ_percentage_change)
            end
            return α, δ
        else
            @check evaluation in [:inf, :overtight, :overlax, :reasonable]
        end

        previous_evaluation = evaluation
        # @logprogress counter/max_iters
    end
    # end # @withprogress
    return α, δ
end

# avoid closures over barrier parameters with a callable struct
# https://discourse.julialang.org/t/function-factories-or-callable-structs/52987
# https://discourse.julialang.org/t/why-is-closure-slower-than-handmade-callable-struct/80361
struct BarrierLosses{T<:Real}
    losses::Function
    controlODE::ControlODE
    α::T
    δ::T
    ρ::T
end

function (l::BarrierLosses)(params)
    l.losses(l.controlODE, params; α=l.α, δ=l.δ, ρ=l.ρ)
end

function constrained_training(
    losses,
    controlODE,
    θ;
    ρ,
    α0=1f-3,
    δ0=1f1,
    max_solver_iterations=10,  # per step
    max_barrier_iterations=100,
    max_α = 1e3 * α0,
    min_δ = 1e-2 * δ0,
    max_δ = 1e4 * δ0,
    show_progressbar=false,
    datadir=nothing,
    metadata=Dict(),  # metadata is added to this dict always
)
    @info "Training with constraints..."

    prog = Progress(max_barrier_iterations;
        desc="Fiacco-McCormick barrier iterations",
        dt=0.2,
        enabled=show_progressbar,
        showspeed=true,
    )

    α, δ = tune_barrier(
        losses,
        controlODE,
        θ;
        ρ,
        α=α0,
        δ=δ0,
        max_iters=100,
        increase_alpha=false,
    )

    α_progression = [α]
    δ_progression = [δ]
    barrier_iteration = 0
    # @withprogress name="Fiacco-McCormick barrier iterations" begin
    while barrier_iteration < max_barrier_iterations
        barrier_iteration += 1

        # @infiltrate any(isinf, (α, δ, ρ))

        # for debugging convenience
        # callable struct to avoid a closure
        lost = BarrierLosses(losses, controlODE, α, δ, ρ)
        # lost(params) = losses(controlODE, params; α, δ, ρ)
        # objective_grad = norm(Zygote.gradient(x -> lost(x)[1], θ)[1])
        # @info "Objective grad" objective_grad
        state_penalty_grad_norm = norm(Zygote.gradient(x -> lost(x)[2], θ)[1])
        @info "State penalty grad norm" state_penalty_grad_norm / norm(θ)
        # regularization_grad = norm(Zygote.gradient(x -> lost(x)[4], θ)[1])
        # @info "Regularization grad" regularization_grad

        # comply with optimization interface using closures
        loss(params) = sum(lost(params))
        # loss(params) = sum(losses(controlODE, params; α, δ, ρ))
        function grad(params)
            try
                return Zygote.gradient(loss, params)[1]
            catch e
                if isa(e, DomainError)
                    return zeros(eltype(params), length(params))
                else
                    # @infiltrate
                    rethrow(e)
                end
            end
        end
        grad!(g, params) = g .= grad(params)

        local minimizer
        let trials = 5, iters = max_solver_iterations
            for trial in 1:trials
                try
                    minimizer = optimize_ipopt(θ, loss, grad!; maxiters=iters)
                    # minimizer = optimize_lbfgsb(θ, loss, grad!)
                    # minimizer = optimize_optim(θ, loss, grad!)
                    break
                catch
                    trial == trials && error("Optimization failed!")
                    # not the smartest workaround...
                    @error "Optimization failed! Retrying with less iterations and added noise... ($trial / $trials)"
                    θ += 1.0f-1 * std(θ) * randn(eltype(θ), length(θ))
                    iters = iters ÷ 2
                end
            end
        end

        objective, state_penalty, control_penalty, regularization = lost(minimizer)

        local_metadata = Dict(
            :iter => barrier_iteration,
            :δ => δ,
            :α => α,
            :ρ => ρ,
            :objective => objective,
            :state_penalty => state_penalty,
            :control_penalty => control_penalty,
            :regularization_cost => regularization,
            :initial_condition => controlODE.u0,
            :tspan => controlODE.tspan,
        )

        !show_progressbar && @info "Barrier iterations" local_metadata
        ProgressMeter.next!(prog; showvalues=[(el.first, el.second) for el in local_metadata])
        # @logprogress barrier_iteration/max_barrier_iterations

        metadata = merge(metadata, local_metadata)
        name = name_interpolation(barrier_iteration)
        store_simulation(name, controlODE, θ; metadata, datadir)

        tuned_α, tuned_δ = tune_barrier(
            losses,
            controlODE,
            minimizer;
            α,
            ρ,
            δ,
            increase_alpha=true,
        )
        if tuned_α > α
            prev_losses = losses(controlODE, minimizer; α, δ, ρ)
            tuned_losses = losses(controlODE, minimizer; α=tuned_α, δ=tuned_δ, ρ)
            @info "Increased alpha" α δ tuned_α tuned_δ prev_losses sum(prev_losses) tuned_losses sum(tuned_losses)
        end
        if (min_δ > tuned_δ > max_δ) || (tuned_α >  max_α)
            @info "Barrier parameter reached bounds." min_δ tuned_δ max_δ tuned_α max_α
            θ = minimizer
            break
        end

        α, δ = tuned_α, tuned_δ
        push!(α_progression, α)
        push!(δ_progression, δ)

        θ = minimizer
    end
    # end # @withprogress
    ProgressMeter.finish!(prog)
    barriers_progression = (; α=α_progression, δ=δ_progression)
    return θ, barriers_progression
end
