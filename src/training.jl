function custom_train!(
    loss_fun,
    p,
    opt::Flux.Optimise.AbstractOptimiser;
    maxiters::Integer=1_000,
    x_tol::Union{Nothing, Real}=nothing,
    f_tol::Union{Nothing, Real}=nothing,
    g_tol::Union{Nothing, Real}=nothing,
    show_progressbar::Bool=true,
    noise_factor=1f-1,
    kwargs...,
)
    @argcheck isnothing(x_tol) || x_tol > zero(x_tol)
    @argcheck isnothing(f_tol) || f_tol > zero(f_tol)

    params = copy(p)
    params_ref = copy(p)

    prog = ProgressUnknown(
        "Flux adaptive training. (f_tol=$f_tol, x_tol=$x_tol, maxiters=$maxiters)";
        enabled=show_progressbar,
        spinner=true,
        showspeed=true,
    )
    iter = 1
    while true
        try
            global loss, back, gradient
            # back is a function that computes the gradient
            loss, back = pullback(loss_fun, params)

            # apply back() to the correct type of 1.0 to get the gradient of the loss.
            gradient = back(one(loss))[1]
        catch
            @warn "Unstable parameter region. Adding some noise to parameters."
            Zygote.@ignore @infiltrate
            params += noise_factor * std(params) * randn(eltype(params), length(params))
            iter += 1
            continue
        end

        # references
        copy!(params_ref, params)

        # optimizer opdate (modifies params)
        Flux.update!(opt, params, gradient)

        # finish metrics
        x_diff = sum(abs2, params_ref - params)
        f_diff = abs2(loss - loss_fun(params))

        # display
        ProgressMeter.next!(
            prog;
            showvalues=[(:iter, iter), (:loss, loss), (:x_diff, x_diff), (:f_diff, f_diff)],
        )

        if !isnothing(x_tol) && x_diff < x_tol
            @info "Space threshold reached: $x_diff < $x_tol tolerance"
            return params
        elseif !isnothing(f_tol) && f_diff < f_tol
            @info "Objective threshold reached: $f_diff < $f_tol tolerance"
            return params
        elseif !isnothing(g_tol)
        elseif iter > maxiters
            @info "Iteration bound reached: $iter > $maxiters tolerance"
            return params
        end
        iter += 1
    end
end

function preconditioner(
    controlODE,
    precondition;
    θ=initial_params(controlODE.controller),
    ρ=nothing,
    saveat=(),
    progressbar=true,
    plot_progress=false,
    plot_final=true,
    # Flux
    optimizer=Optimiser(WeightDecay(1f-4), ADAM(1f-2)),
    # SciML
    # optimizer=LBFGS(; linesearch=BackTracking()),
    # optimizer=optimizer_with_attributes(
    #     Ipopt.Optimizer,
    #     "print_level" => 3,
    #     "tol" => 1e-1,
    #     "max_iter" => 20,
    # ),
    # adtype=GalacticOptim.AutoZygote(),
    integrator=INTEGRATOR,
    sensealg=SENSEALG,
    kwargs...,
)
    @info "Preconditioning..."

    prog = Progress(
        length(controlODE.tsteps[2:end]);
        desc="Pretraining in subintervals...",
        dt=0.5,
        showspeed=true,
        enabled=progressbar,
    )
    for partial_time in controlODE.tsteps[2:(end - 1)]  # skip spurious ends from collocation
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
        fixed_sol = solve(fixed_prob, integrator; saveat, sensealg)

        # Zygote ignore anything unrelated to loss function
        function precondition_loss(params; plot=nothing)
            plot_arrays = Dict(:reference => [], :control => [])

            sum_squares = 0.0f0

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
                        for r in 1:size(reference, 1)
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
            if isnothing(ρ)
                return sum_squares
            else
                regularization = ρ * sum(abs2, params)
                return sum_squares + regularization
            end
        end

        # opt_result = sciml_train(precondition_loss, θ, optimizer, adtype; kwargs...)
        # θ = opt_result.minimizer

        θ = custom_train!(precondition_loss, θ, optimizer; kwargs...)

        pvar = plot_progress ? :unicode : nothing
        ploss = precondition_loss(θ; plot=pvar)
        ProgressMeter.next!(prog; showvalues=[(:loss, ploss)])
        if partial_time == controlODE.tsteps[end] && plot_final
            precondition_loss(θ; plot=:pyplot)
        end
    end
    return θ
end

function constrained_training(
    controlODE,
    losses,
    δ0;
    α,
    ρ,
    θ=initial_params(controlODE.controller),
    optimizer,
    # Flux
    # optimizer=Optimiser(WeightDecay(1f-4), ADAM(1f-2)),
    # Optim
    # optimizer=LBFGS(; linesearch=BackTracking()),
    # SciML
    # optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 3, "tol" => 1e-2, "max_iter" =>100),
    # adtype=GalacticOptim.AutoZygote(),
    sensealg=SENSEALG,
    max_barrier_iterations=50,
    δ_final=1.0f-1 * δ0,
    loosen_rule=(δ) -> (1.05f0 * δ),
    tighten_rule=(δ) -> (0.9f0 * δ),
    state_penalty_ratio_limit=1f3,
    show_progressbar=false,
    datadir=nothing,
    metadata=Dict(),  # metadata is added to this dict always
    kwargs...,
)
    @info "Training with constraints..."

    prog = ProgressUnknown(
        "Fiacco-McCormick barrier iterations";
        enabled=show_progressbar,
        spinner=true,
        showspeed=true,
    )

    δ = δ0
    δ_progression = []
    counter = 1
    while δ > δ_final && counter <= max_barrier_iterations
        counter += 1

        # for debugging convenience
        lost(params) = losses(controlODE, params; α, δ, ρ, sensealg)

        # closures to comply with optimization interface
        loss(params) = sum(losses(controlODE, params; α, δ, ρ, sensealg))
        grad(params) = Zygote.gradient(loss, params)[1]
        # grad!(G, params) = G .= Zygote.gradient(loss, params)[1]

        # Optim
        # https://julianlsolvers.github.io/Optim.jl/stable/#user/config/
        options = Optim.Options(store_trace=true, show_trace=true, extended_trace=false, kwargs...)
        result = Optim.optimize(loss, grad, θ, optimizer, options; inplace=false)
        minimizer = result.minimizer

        # Flux
        # minimizer = custom_train!(loss, θ, optimizer; controlODE, losses, α, δ, sensealg, kwargs...)

        Zygote.@ignore @infiltrate eltype(minimizer) != eltype(θ)

        current_losses = losses(controlODE, minimizer; α, δ, ρ)
        objective, state_penalty, control_penalty, regularization = current_losses

        current_values = [
            (:iter, counter),
            (:α, α),
            (:δ, δ),
            (:objective, objective),
            (:state_penalty, state_penalty),
            # (:control_penalty, control_penalty),
            (:regularization, regularization),
        ]
        @info "Fiacco-McCormick barrier iterations" current_values
        ProgressMeter.next!(
            prog;
            showvalues=current_values,
        )

        local_metadata = Dict(
            :parameters => θ,
            :δ => δ,
            :α => α,
            :objective => objective,
            :state_penalty => state_penalty,
            :control_penalty => control_penalty,
            :regularization_cost => regularization,
            :count_params => length(initial_params(controlODE.controller)),
            :layers => controller_shape(controlODE.controller),
            :tspan => controlODE.tspan,
            :tsteps => controlODE.tsteps,
        )
        metadata = merge(metadata, local_metadata)
        store_simulation(
            "delta_$(round(δ, digits=2))_iter_$counter", controlODE, θ; metadata, datadir
        )

        state_penalty_size = abs(state_penalty)
        other_penalties_size = abs(objective)
        # other_penalties_size = max(abs(objective), abs(control_penalty), abs(regularization))
        state_penalty_ratio = state_penalty_size / other_penalties_size
        if any(isinf, current_losses) || state_penalty_ratio > state_penalty_ratio_limit
            δ = loosen_rule(δ)
            push!(δ_progression, δ)
            Zygote.@ignore @infiltrate
            continue
        end
        δ = tighten_rule(δ)
        push!(δ_progression, δ)

        Zygote.@ignore @infiltrate counter % 10 == 0
        θ = minimizer

        # add some noise to avoid local minima
        # θ += 1.0f-1 * std(θ) * randn(Float32, length(θ))
    end
    Zygote.@ignore @infiltrate
    return θ, δ
end
