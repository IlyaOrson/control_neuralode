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

function optimize_nlopt(θ, loss, grad!; algorithm=:LD_MMA, xtol_rel=1e-2)
    # gradient based algorithms available
    # https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
    reasonable_algs = [:LD_MMA, :LD_CCSAQ, :LD_SLSQP, :LD_LBFGS, :LD_TNEWTON_PRECOND_RESTART]
    @argcheck algorithm in reasonable_algs
    params_size=length(θ)
    opt = NLopt.Opt(algorithm, length(θ))
    opt.lower_bounds = fill(-1e1, params_size)
    opt.upper_bounds = fill(1e1, params_size)
    opt.xtol_rel = xtol_rel

    function loss_grad(x, g)
        grad!(g, x)
        return loss(x)
    end
    opt.min_objective = loss_grad

    (minf, minx, ret) = NLopt.optimize(opt, θ)
    if ret ∈ [:FORCED_STOP, :ROUNDOFF_LIMITED, :OUT_OF_MEMORY, :INVALID_ARGS, :FAILURE]
        @error "Optimization failed with status $ret"
        return nothing
    else
        @info "Found minimum $minf after $(opt.numevals) iterations (returned $ret)."
    end
    return minx
end

function optimize_ipopt(
    θ,
    loss,
    grad!;
    maxiters::Int=1_000,
    tolerance::Float64=1e-2,
    verbosity::Union{Int, Nothing}=3
)
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
    if !isnothing(verbosity)
        Ipopt.AddIpoptIntOption(ipopt, "print_level", verbosity)  # default is 5
    end
    Ipopt.AddIpoptNumOption(ipopt, "tol", tolerance)
    Ipopt.AddIpoptIntOption(ipopt, "max_iter", maxiters)
    Ipopt.AddIpoptNumOption(ipopt, "acceptable_tol", 0.1*tolerance)  # default is 1e-6
    Ipopt.AddIpoptIntOption(ipopt, "acceptable_iter", 5)  # default is 15
    Ipopt.AddIpoptStrOption(ipopt, "check_derivatives_for_naninf", "yes")
    Ipopt.AddIpoptStrOption(ipopt, "print_info_string", "yes")
    Ipopt.AddIpoptStrOption(ipopt, "hessian_approximation", "limited-memory")
    Ipopt.AddIpoptStrOption(ipopt, "mu_strategy", "adaptive")
    # A (slow!) derivative test will be performed before the optimization.
    # The test is performed at the user provided starting point and marks derivative values that seem suspicious
    Ipopt.AddIpoptStrOption(ipopt, "derivative_test", "first-order")

    # https://github.com/jump-dev/Ipopt.jl/blob/master/src/MOI_wrapper.jl#L1261
    local solve_status
    optimizer_output = @capture_out begin
        solve_status = Ipopt.IpoptSolve(ipopt)
    end
    if isnothing(verbosity)
        # only first entry of logger gets formatted as a string
        @info "Ipopt report" status=Ipopt._STATUS_CODES[solve_status]
    elseif verbosity != 0
        @info "Ipopt report (verbosity=$(verbosity))\n" * optimizer_output
    end
    # @infiltrate !in(solve_status, [0, 1, -1, -3])   # debugging
    return ipopt.x  # ipopt_minimizer
end
