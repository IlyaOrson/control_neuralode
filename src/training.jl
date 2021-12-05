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
    plot_progress=false,
    plot_final=true,
)
    @assert t0 < time_fractions[1]
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
        fixed_sol = OrdinaryDiffEq.solve(
            fixed_prob, BS3(); abstol=1f-1, reltol=1f-1, saveat
        )

        function precondition_loss(params; plot=nothing)
            plot_arrays = Dict(:reference => [], :control => [])
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
                    if !isnothing(plot)
                        push!(plot_arrays[:reference], reference)
                        push!(plot_arrays[:control], control)
                    end
                end
            end
            Zygote.ignore() do
                if !isnothing(plot)
                    @assert plot in [:unicode, :pyplot]
                    reference = reduce(hcat, plot_arrays[:reference])
                    control = reduce(hcat, plot_arrays[:control])
                    @assert length(reference) == length(control)
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
        end

        preconditioner = DiffEqFlux.sciml_train(
            precondition_loss,
            θ,
            BFGS(; initial_stepnorm=0.01);
            # maxiters=10,
            allow_f_increases=true,
            f_tol,
        )
        θ = preconditioner.minimizer
        pvar = plot_progress ? :unicode : nothing
        ploss = precondition_loss(θ; plot=pvar)
        next!(prog; showvalues=[(:loss, ploss)])
        if partial_time == time_fractions[end] && plot_final
            precondition_loss(θ; plot=:pyplot)
        end
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
    show_progressbar=false,
    plots_callback=nothing,
    f_tol=1f-1,
    datadir=nothing,
    metadata=Dict(),  # metadata is added to this dict always
)
    @assert length(αs) == length(δs)
    prog = Progress(
        length(αs); desc="Training with constraints...", enabled=show_progressbar
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
            LBFGS(; linesearch=BackTracking());
            # cb=print_callback,
            allow_f_increases=true,
            f_tol,
        )
        θ = result.minimizer

        objective, state_penalty, control_penalty, regularization = loss(
            θ, prob; α, δ, tsteps
        )

        @info "Current values" α,
        δ, objective, state_penalty, control_penalty,
        regularization

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
            "delta_$(round(δ, digits=2))", controller, prob, θ, tsteps; metadata, datadir
        )
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
