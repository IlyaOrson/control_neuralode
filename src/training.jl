function optimize_flux(
    θ,
    loss;
    opt::Flux.Optimise.AbstractOptimiser=ADAMW(0.01, (0.9, 0.999), 1.0f-2),
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

    params = copy(θ)
    params_ref = copy(θ)

    prog = Progress(maxiters;
        desc="Flux adaptive training. (x_tol=$x_tol, f_tol=$f_tol, g_tol=$g_tol, maxiters=$maxiters)",
        dt=0.2,
        enabled=show_progressbar,
        showspeed=true,
        offset=1,
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
        current_values = [
            (:iter, iter),
            (:loss, current_loss),
            (:x_diff, x_diff),
            (:f_diff, f_diff),
            (:g_norm, g_norm),
        ]
        ProgressMeter.next!(prog; showvalues=current_values)

        if !isnothing(x_tol) && x_diff < x_tol
            desc = "Space rate threshold reached: $x_diff < $x_tol tolerance"
            @debug desc
            # ProgressMeter.finish!(prog; desc)
            return params
        elseif !isnothing(f_tol) && f_diff < f_tol
            desc = "Objective rate threshold reached: $f_diff < $f_tol tolerance"
            @debug desc
            # ProgressMeter.finish!(prog; desc)
            return params
        elseif !isnothing(g_tol) && g_norm < g_tol
            desc = "Gradient norm threshold reached: $g_norm < $g_tol tolerance"
            @debug desc
            # ProgressMeter.finish!(prog; desc)
            return params
        elseif iter > maxiters
            desc = "Iteration bound reached: $iter > $maxiters tolerance"
            @debug desc
            # ProgressMeter.finish!(prog; desc)
            return params
        end
        iter += 1
    end
end

function optimize_lbfgsb(θ, loss, grad!)
    # LBFGSB
    # https://github.com/Gnimuc/LBFGSB.jl/blob/master/test/wrapper.jl
    params_size = length(θ)
    lbfgsb = LBFGSB.L_BFGS_B(params_size, 10)
    bounds = zeros(3, params_size)
    for i in 1:params_size
        bounds[1, i] = 2  # 0->unbounded, 1->only lower bound, 2-> both lower and upper bounds, 3->only upper bound
        bounds[2, i] = -1e1
        bounds[3, i] = 1e1
    end
    fout, xout = lbfgsb(
        loss,
        grad!,
        Float64.(θ),
        bounds;
        m=5,
        factr=1e7,
        pgtol=1e-5,
        iprint=-1,
        maxfun=1000,
        maxiter=100,
    )
    return xout
end

function optimize_optim(θ, loss, grad!)
    # Optim
    # https://julianlsolvers.github.io/Optim.jl/stable/#user/config/
    optim_options = Optim.Options(;
        store_trace=true, show_trace=false, extended_trace=false
    )
    # optim_result = Optim.optimize(loss, grad!, θ, BFGS(), optim_options)
    optim_result = Optim.optimize(loss, grad!, θ, LBFGS(; linesearch=BackTracking()), optim_options)
    return optim_result.minimizer
end

function optimize_ipopt(θ, loss, grad!; maxiters::Int=1_000, tolerance::Float64=1e-2, verbosity::Int=3)
    # IPOPT
    # https://github.com/jump-dev/Ipopt.jl/blob/master/test/C_wrapper.jl
    eval_g(x, g) = g[:] = zero(x)
    eval_grad_f(x, g) = grad!(g, x)
    eval_jac_g(x, rows, cols, values) = return nothing
    # eval_h(x, rows, cols, obj_factor, lambda, values) = return nothing
    params_size=length(θ)
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
    Ipopt.AddIpoptIntOption(ipopt, "print_level", verbosity)  # default is 5
    Ipopt.AddIpoptNumOption(ipopt, "tol", tolerance)
    Ipopt.AddIpoptIntOption(ipopt, "max_iter", maxiters)
    Ipopt.AddIpoptNumOption(ipopt, "acceptable_tol", 1e-1)  # default is 1e-6
    Ipopt.AddIpoptIntOption(ipopt, "acceptable_iter", 5)  # default is 15
    Ipopt.AddIpoptStrOption(ipopt, "check_derivatives_for_naninf", "yes")
    Ipopt.AddIpoptStrOption(ipopt, "print_info_string", "yes")
    Ipopt.AddIpoptStrOption(ipopt, "hessian_approximation", "limited-memory")
    Ipopt.AddIpoptStrOption(ipopt, "mu_strategy", "adaptive")

    # https://github.com/jump-dev/Ipopt.jl/blob/master/src/MOI_wrapper.jl#L1261
    solve_status = Ipopt.IpoptSolve(ipopt)
    @info "Ipopt result" status=Ipopt._STATUS_CODES[solve_status]
    return ipopt.x  # ipopt_minimizer
end

function preconditioner(
    controlODE,
    precondition;
    θ,
    ρ=nothing,
    saveat=(),
    progressbar=false,
    plot_final=true,
    integrator=INTEGRATOR,
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
        θ = optimize_flux(θ, precondition_loss; kwargs...)
        # θ = optimize_ipopt(θ, precondition_loss, grad!)

        ProgressMeter.next!(prog)
        if partial_time == controlODE.tsteps[end] && plot_final
            precondition_loss(θ; plot=:pyplot)
        end
    end
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
    α_percentage_change=25f0,
    max_iters=100,
    increase_alpha=true,
    verbose=false,
)
    previous_evaluation = :reasonable
    counter=0

    while true
        counter+=1
        if counter > max_iters
            return α, δ
        end

        evaluation = evaluate_barrier(losses, controlODE, θ; α, δ, ρ, verbose)

        if evaluation == :overlax
            δ = decrease_by_percentage(δ, δ_percentage_change)

        elseif evaluation in [:overtight, :inf]
            δ = increase_by_percentage(δ, δ_percentage_change)

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
    end
end

function constrained_training(
    losses,
    controlODE,
    θ;
    ρ,
    α0=1f-2,
    δ0=1f1,
    max_barrier_iterations=50,
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
        increase_alpha=false,
    )

    # initial_α = α
    # initial_δ = δ

    α_progression = [α]
    δ_progression = [δ]
    barrier_iteration = 0
    while barrier_iteration < max_barrier_iterations
        barrier_iteration += 1

        # for debugging convenience
        lost(params) = losses(controlODE, params; α, δ, ρ)
        # objective_grad = sum(abs2, Zygote.gradient(x -> lost(x)[1], θ)[1])
        # state_penalty_grad = sum(abs2, Zygote.gradient(x -> lost(x)[2], θ)[1])
        # regularization_grad = sum(abs2, Zygote.gradient(x -> lost(x)[4], θ)[1])
        # @info "Objective grad" objective_grad
        # @info "State penalty grad" state_penalty_grad
        # @info "Regularization grad" regularization_grad

        # closures to comply with optimization interface
        loss(params) = sum(losses(controlODE, params; α, δ, ρ))
        grad(params) = Zygote.gradient(loss, params)[1]
        grad!(g, params) = g .= Zygote.gradient(loss, params)[1]

        local minimizer
        optimizer_output = @capture_out begin
            # minimizer = optimize_flux(θ, loss)
            minimizer = optimize_ipopt(θ, loss, grad!)
            # minimizer = optimize_optim(θ, loss, grad!)
        end
        # needs to be print to format the output as originally intended
        # println(optimizer_output)

        objective, state_penalty, control_penalty, regularization = lost(minimizer)

        current_values = [
            (:iter, barrier_iteration),
            (:α, α),
            (:δ, δ),
            (:ρ, ρ),
            (:objective, objective),
            (:state_penalty, state_penalty),
            (:control_penalty, control_penalty),
            (:regularization, regularization),
        ]
        # @info "Barrier iterations" current_values
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
            # :layers => controller_shape(controlODE.controller),
            :tspan => controlODE.tspan,
        )
        metadata = merge(metadata, local_metadata)
        name = name_interpolation(δ, barrier_iteration)
        store_simulation(name, controlODE, θ; metadata, datadir)

        # local tuned_α, tuned_δ
        # for tuning_percentage in [δ_percentage_change * (1/2^i) for i in 0:4]
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
                @show α, δ, tuned_α, tuned_δ
                @show losses(controlODE, minimizer; α, δ, ρ)
                @show losses(controlODE, minimizer; α=tuned_α, δ=tuned_δ, ρ)
            end
        #     if tuned_δ < δ
        #         break
        #     end
        # end

        α, δ = tuned_α, tuned_δ
        push!(α_progression, α)
        push!(δ_progression, δ)

        # add some noise to avoid local minima
        # θ += 1.0f-1 * std(θ) * randn(Float32, length(θ))

        θ = minimizer
    end
    ProgressMeter.finish!(prog)
    barriers_progression = (; α=α_progression, δ=δ_progression)
    return θ, barriers_progression
end
