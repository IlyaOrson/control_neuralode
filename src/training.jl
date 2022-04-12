function preconditioner(
    controlODE,
    precondition;
    θ=initial_params(controlODE.controller),
    reg_coeff=1.0f0,
    saveat=(),
    progressbar=true,
    plot_progress=false,
    plot_final=true,
    optimizer=LBFGS(; linesearch=BackTracking()),
    # optimizer=optimizer_with_attributes(
    #     Ipopt.Optimizer, "print_level" => 3, "tol" => 1e-1, "max_iter" => 20
    # ),
    integrator=INTEGRATOR,
    sensealg=SENSEALG,
    adtype=GalacticOptim.AutoZygote(),
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
    for partial_time in controlODE.tsteps[2:end]
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
            # mean_squares = 0.0f0

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
            regularization = reg_coeff * mean(abs2, params)
            return sum_squares + regularization
            # return sum_squares / mean_squares + regularization
        end

        optimization = sciml_train(precondition_loss, θ, optimizer, adtype; kwargs...)
        θ = optimization.minimizer
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
    α0,
    δ0;
    θ=initial_params(controlODE.controller),
    show_progressbar=false,
    plots_callback=nothing,
    datadir=nothing,
    metadata=Dict(),  # metadata is added to this dict always
    optimizer=LBFGS(; linesearch=BackTracking()), # optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 3, "tol" => 1e-2, "max_iter" =>100),
    sensealg=SENSEALG,
    adtype=GalacticOptim.AutoZygote(),
    max_barrier_iterations=10,
    δ_final=1.0f-1 * δ,
    loosen_rule=(δ) -> (1.2f0 * δ),
    tighten_rule=(δ) -> (0.7f0 * δ),
    kwargs...,
)
    @info "Training with constraints..."

    prog = ProgressUnknown(
        "Fiacco-McCormick barrier iterations";
        enabled=show_progressbar,
        spinner=true,
        showspeed=true,
    )

    α = α0
    δ = δ0

    counter = 1
    while δ > δ_final || counter <= max_barrier_iterations
        counter += 1

        # closure to comply with optimization interface
        loss(params) = sum(losses(controlODE, params; α, δ, sensealg, kwargs...))

        # if !isnothing(plots_callback)
        #     plots_callback(controlODE, θ)
        # end

        # function print_callback(params, loss)
        #     println(loss)
        #     return false
        # end

        result = sciml_train(loss, θ, optimizer, adtype; kwargs...)

        current_losses = losses(controlODE, result.minimizer; α, δ, kwargs...)
        objective, state_penalty, control_penalty, regularization = current_losses

        ProgressMeter.next!(
            prog;
            showvalues=[
                (:α, α),
                (:δ, δ),
                (:objective, objective),
                (:state_penalty, state_penalty),
                (:control_penalty, control_penalty),
                (:regularization, regularization),
            ],
        )

        if any(isinf, current_losses)
            δ = loosen_rule(δ)
            continue
        end
        δ = tighten_rule(δ)
        θ = result.minimizer

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

        # add some noise to avoid local minima
        θ += 1.0f-1 * std(θ) * randn(length(θ))
    end
    return θ, δ
end
