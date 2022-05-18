function flux_train!(
    loss,
    p,
    opt::Flux.Optimise.AbstractOptimiser;
    maxiters::Integer=1_000,
    x_tol::Union{Nothing,Real}=nothing,
    f_tol::Union{Nothing,Real}=nothing,
    g_tol::Union{Nothing,Real}=nothing,
    show_progressbar::Bool=true,
    noise_factor=1.0f-1,
    kwargs...,
)
    @argcheck isnothing(x_tol) || x_tol > zero(x_tol)
    @argcheck isnothing(f_tol) || f_tol > zero(f_tol)
    @argcheck isnothing(g_tol) || g_tol > zero(g_tol)

    params = copy(p)
    params_ref = copy(p)

    prog = ProgressUnknown(
        "Flux adaptive training. (x_tol=$x_tol, f_tol=$f_tol, g_tol=$g_tol, maxiters=$maxiters)";
        enabled=show_progressbar,
        spinner=true,
        showspeed=true,
    )
    iter = 1
    while true
        local current_loss, back, gradient
        try
            # global current_loss, back, gradient
            # back is a function that computes the gradient
            current_loss, back = pullback(loss, params)

            # apply back() to the correct type of 1.0 to get the gradient of the loss.
            gradient = back(one(current_loss))[1]
        catch
            @warn "Unstable parameter region. Adding some noise to parameters."
            # Zygote.@ignore @infiltrate
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
        f_diff = abs2(current_loss - loss(params))
        g_norm = sum(abs2, gradient)

        # display
        ProgressMeter.next!(
            prog;
            showvalues=[
                (:iter, iter),
                (:loss, current_loss),
                (:x_diff, x_diff),
                (:f_diff, f_diff),
                (:g_norm, g_norm),
            ],
        )

        if !isnothing(x_tol) && x_diff < x_tol
            @info "Space rate threshold reached: $x_diff < $x_tol tolerance"
            return params
        elseif !isnothing(f_tol) && f_diff < f_tol
            @info "Objective rate threshold reached: $f_diff < $f_tol tolerance"
            return params
        elseif !isnothing(g_tol) && g_norm < g_tol
            @info "Gradient norm threshold reached: $g_norm < $g_tol tolerance"
            return params
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
    θ,
    ρ=nothing,
    saveat=(),
    progressbar=true,
    plot_progress=false,
    plot_final=true,
    # Flux
    optimizer=Optimiser(WeightDecay(1.0f-2), ADAM(1.0f-2)),
    # Optim
    # optimizer=LBFGS(; linesearch=BackTracking()),
    integrator=INTEGRATOR,
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
        fixed_sol = solve(fixed_prob, integrator; saveat)

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

        θ = flux_train!(precondition_loss, θ, optimizer; kwargs...)

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
    θ,
    max_barrier_iterations=50,
    δ_final=1.0f-1 * δ0,
    loosen_rule=(δ) -> (1.025f0 * δ),
    tighten_rule=(δ) -> (0.95f0 * δ),
    state_penalty_ratio_limit=1.0f3,
    show_progressbar=false,
    datadir=nothing,
    metadata=Dict(),  # metadata is added to this dict always
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
        lost(params) = losses(controlODE, params; α, δ, ρ)
        objective_grad = sum(abs2, Zygote.gradient(x -> lost(x)[1], θ)[1])
        state_penalty_grad = sum(abs2, Zygote.gradient(x -> lost(x)[2], θ)[1])
        regularization_grad = sum(abs2, Zygote.gradient(x -> lost(x)[4], θ)[1])
        @info "Objective grad" objective_grad
        @info "State penalty grad" state_penalty_grad
        @info "Regularization grad" regularization_grad

        # closures to comply with optimization interface
        params_size = length(θ)
        loss(params) = sum(losses(controlODE, params; α, δ, ρ))
        grad(params) = Zygote.gradient(loss, params)[1]
        grad!(g, params) = g .= Zygote.gradient(loss, params)[1]

        # LBFGSB
        # https://github.com/Gnimuc/LBFGSB.jl/blob/master/test/wrapper.jl
        # lbfgsb = LBFGSB.L_BFGS_B(params_size, 10)
        # bounds = zeros(3, params_size)
        # for i in 1:params_size
        #     bounds[1, i] = 2  # 0->unbounded, 1->only lower bound, 2-> both lower and upper bounds, 3->only upper bound
        #     bounds[2, i] = -1e1
        #     bounds[3, i] = 1e1
        # end
        # fout, xout = lbfgsb(
        #     loss,
        #     grad!,
        #     Float64.(θ),
        #     bounds;
        #     m=5,
        #     factr=1e7,
        #     pgtol=1e-5,
        #     iprint=-1,
        #     maxfun=1000,
        #     maxiter=100,
        # )

        # IPOPT
        # https://github.com/jump-dev/Ipopt.jl/blob/master/test/C_wrapper.jl
        eval_g(x, g) = g[:] = zero(x)
        eval_grad_f(x, g) = grad!(g, x)
        eval_jac_g(x, rows, cols, values) = return nothing
        # eval_h(x, rows, cols, obj_factor, lambda, values) = return nothing
        x_lb = fill(-1e1, params_size)
        x_ub = fill(1e1, params_size)
        ipopt = Ipopt.CreateIpoptProblem(
            params_size,
            x_lb,
            x_ub,
            0,
            Float64[],
            Float64[],
            0,
            0,
            loss,
            eval_g,
            eval_grad_f,
            eval_jac_g,
            nothing,
        )
        ipopt.x = Float64.(θ)
        Ipopt.AddIpoptIntOption(ipopt, "print_level", 5)
        Ipopt.AddIpoptIntOption(ipopt, "max_iter", 200)
        Ipopt.AddIpoptStrOption(ipopt, "check_derivatives_for_naninf", "yes")
        Ipopt.AddIpoptStrOption(ipopt, "print_info_string", "yes")
        Ipopt.AddIpoptStrOption(ipopt, "hessian_approximation", "limited-memory")
        Ipopt.AddIpoptStrOption(ipopt, "mu_strategy", "adaptive")
        Ipopt.AddIpoptNumOption(ipopt, "tol", 1e-2)

        # https://github.com/jump-dev/Ipopt.jl/blob/d9e9176620a9b527a08991a3d41062fa948867f7/src/Ipopt.jl#L113
        solve_status = Ipopt.IpoptSolve(ipopt)
        ipopt_minimizer = ipopt.x

        # Optim
        # https://julianlsolvers.github.io/Optim.jl/stable/#user/config/
        optim_options = Optim.Options(;
            store_trace=true, show_trace=true, extended_trace=false
            )
        # optim_result = Optim.optimize(loss, grad!, θ, BFGS(), optim_options)
        optim_result = Optim.optimize(loss, grad!, θ, LBFGS(; linesearch=BackTracking()), optim_options)

        # Flux
        # flux_minimizer = flux_train!(loss, θ, ADAM())

        # @info "LBFGSB" lost(xout)
        @info "IPOPT" lost(ipopt_minimizer)
        @info "Optim" lost(optim_result.minimizer)
        # @info "Flux" lost(flux_minimizer)
        # Zygote.@ignore @infiltrate

        minimizer = ipopt.x
        objective, state_penalty, control_penalty, regularization = lost(minimizer)

        current_values = [
            (:iter, counter),
            (:α, α),
            (:δ, δ),
            (:ρ, ρ),
            (:objective, objective),
            (:state_penalty, state_penalty),
            (:control_penalty, control_penalty),
            (:regularization, regularization),
        ]
        # @info "Fiacco-McCormick barrier iterations" current_values
        ProgressMeter.next!(prog; showvalues=current_values)

        local_metadata = Dict(
            :δ => δ,
            :α => α,
            :ρ => ρ,
            :objective => objective,
            :state_penalty => state_penalty,
            :control_penalty => control_penalty,
            :regularization_cost => regularization,
            :num_params => length(initial_params(controlODE.controller)),
            :layers => controller_shape(controlODE.controller),
            :tspan => controlODE.tspan,
            :tsteps => controlODE.tsteps,
        )
        metadata = merge(metadata, local_metadata)
        name = name_interpolation(δ, counter)
        store_simulation(name, controlODE, θ; metadata, datadir)

        state_penalty_size = abs(state_penalty)
        other_penalties_size = abs(objective)
        # other_penalties_size = max(abs(objective), abs(control_penalty), abs(regularization))
        state_penalty_ratio = state_penalty_size / other_penalties_size
        if any(isinf, lost(minimizer)) || state_penalty_ratio > state_penalty_ratio_limit
            δ = loosen_rule(δ)
            push!(δ_progression, δ)
            # Zygote.@ignore @infiltrate
            continue
        end
        δ = tighten_rule(δ)
        push!(δ_progression, δ)

        θ = minimizer

        # add some noise to avoid local minima
        # θ += 1.0f-1 * std(θ) * randn(Float32, length(θ))
    end
    # Zygote.@ignore @infiltrate
    return θ, δ
end
