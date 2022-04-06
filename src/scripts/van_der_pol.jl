# Vassiliadis, V. S., Sargent, R. W. H., & Pantelides, C. C. (1994).
# Solution of a Class of Multistage Dynamic Optimization Problems. 2. Problems with Path Constraints.
# Industrial & Engineering Chemistry Research, 33(9), 2123–2133. https://doi.org/10.1021/ie00033a015

function van_der_pol(; store_results=false::Bool)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    # initial conditions and timepoints
    t0 = 0.0f0
    tf = 5.0f0
    u0 = [0.0f0, 1.0f0]
    tspan = (t0, tf)
    Δt = 0.1f0

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(2, 12, tanh_fast),
        FastDense(12, 12, tanh_fast),
        FastDense(12, 1),
        (x, p) -> (1.3f0 .* sigmoid_fast.(x)) .- 0.3f0,
    )

    system = VanDerPol()
    controlODE = ControlODE(controller, system, u0, tspan; Δt)

    θ = initial_params(controlODE.controller)

    _, states_raw, _ = run_simulation(controlODE, θ)
    phase_portrait(
        controlODE,
        θ,
        square_bounds(u0, 7);
        projection=[1, 2],
        markers=states_markers(states_raw),
        title="Initial policy",
    )

    collocation = van_der_pol_collocation(
        controlODE.u0,
        controlODE.tspan;
        num_supports=length(controlODE.tsteps),
        nodes_per_element=2,
        constrain_states=false,
    )
    reference_controller = interpolant_controller(collocation; plot=false)

    θ = preconditioner(
        controlODE,
        reference_controller;
        #control_range_scaling=[maximum(collcation_results.controls) - minimum(collcation_results.controls)],
    )

    plot_simulation(controlODE, θ; only=:controls)
    store_simulation("precondition", controlODE, θ; datadir)

    _, states_raw, _ = run_simulation(controlODE, θ)
    phase_portrait(
        controlODE,
        θ,
        square_bounds(u0, 3);
        projection=[1, 2],
        markers=states_markers(states_raw),
        title="Preconditioned policy",
    )

    # closures to comply with required interface
    function plotting_callback(params, loss)
        return plot_simulation(controlODE, params; only=:states, vars=[1], show=loss)
    end

    ### define objective function to optimize
    function loss(controlODE, params; kwargs...)
        sol = solve(controlODE, params; kwargs...) |> Array
        # return Array(sol)[3, end]  # return last value of third variable ...to be minimized
        sum_squared = 0.0f0
        for i in 1:size(sol, 2)
            s = sol[:, i]
            c = controlODE.controller(s, params)
            sum_squared += s[1]^2 + s[2]^2 + c[1]^2  # TODO: use relaxed logarithm
        end
        return sum_squared
    end
    loss(params) = loss(controlODE, params)

    @info "Training..."
    optimizer = LBFGS(; linesearch=BackTracking())
    # optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 3, "tol" => 1e-2, "max_iter"=>20)
    result = sciml_train(
        loss,
        θ,
        optimizer;
        # cb=plotting_callback,
        # allow_f_increases=true,
    )
    θ = result.minimizer

    _, states_raw, _ = run_simulation(controlODE, θ)
    phase_portrait(
        controlODE,
        θ,
        square_bounds(u0, 3);
        projection=[1, 2],
        markers=states_markers(states_raw),
        title="Optimized policy",
    )

    store_simulation(
        "unconstrained",
        controlODE,
        θ;
        datadir,
        metadata=Dict(:loss => loss(θ), :constraint => "none"),
    )

    ### now add state constraint x1(t) > -0.4 with
    function penalty_loss(controlODE, params; α=10.0f0, kwargs...)
        # integrate ODE system
        sol = solve(controlODE, params; kwargs...) |> Array
        sum_squared = 0.0f0
        for i in 1:size(sol, 2)
            s = sol[:, i]
            c = controlODE.controller(s, params)
            sum_squared += s[1]^2 + s[2]^2 + c[1]^2
        end
        fault = min.(sol[1, 1:end] .+ 0.4f0, 0.0f0)
        penalty = α * sum(fault .^ 2)  # quadratic penalty
        return sum_squared + penalty
    end

    @info "Enforcing constraints..."
    penalty_coefficients = [1.0f1, 1.0f2, 1.0f3, 1.0f4]
    prog = Progress(
        length(penalty_coefficients);
        desc="Fiacco-McCormick iterations",
        dt=0.5,
        showspeed=true,
        enabled=true,
    )
    for α in penalty_coefficients

        # function plotting_callback(params, loss)
        #     return plot_simulation(
        #         controlODE, params; only=:states, vars=[1], show=loss
        #     )
        # end

        # closures to comply with interface
        penalty_loss_(params) = penalty_loss(controlODE, params; α)

        # plot_simulation(
        #     controlODE,
        #     θ;
        #     only=:controls,
        #     show=penalty_loss_(θ),
        # )

        optimizer = LBFGS(; linesearch=BackTracking());
        # optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 3, "tol" => 1e-2, "max_iter" => 100)
        result = sciml_train(
            penalty_loss_,
            θ,
            optimizer,
            # iterations=50,
            # allow_f_increases=true,
            # cb=plotting_callback,
        )
        θ = result.minimizer
        next!(prog; showvalues=[(:α, α), (:loss, penalty_loss_(θ))])
    end
    θ = result.minimizer
    optimal = result.minimum

    # penalty_loss(result.minimizer, constrained_prob, tsteps; α=penalty_coefficients[end])
    plot_simulation(controlODE, θ; only=:controls)

    store_simulation(
        "constrained",
        controlODE,
        θ;
        datadir,
        metadata=Dict(
            :loss => penalty_loss(controlODE, θ; α=penalty_coefficients[end]),
            :constraint => "quadratic x2(t) > -0.4",
        ),
    )

    _, states_opt, _ = run_simulation(controlODE, θ)
    function indicator(coords...)
        if coords[1] > -0.4
            return true
        end
        return false
    end
    shader = ShadeConf(; indicator)
    phase_portrait(
        controlODE,
        θ,
        square_bounds(u0, 3);
        shader,
        projection=[1, 2],
        markers=states_markers(states_opt),
        title="Optimized policy with constraints",
    )

    # u0 = [0f0, 1f0]
    perturbation_specs = [
        (variable=1, type=:positive, scale=1.0f0, samples=3, percentage=1.0f-1)
        (variable=2, type=:negative, scale=1.0f0, samples=3, percentage=1.0f-1)
        # (variable=3, type=:positive, scale=20.0f0, samples=8, percentage=2.0f-2)
    ]
    constraint_spec = ConstRef(; val=-0.4, direction=:horizontal, class=:state, var=1)

    # plot_initial_perturbations_collocation(
    #     controlODE,
    #     θ,
    #     perturbation_specs,
    #     van_der_pol_collocation;
    #     refs=[constraint_spec],
    #     storedir=datadir,
    # )
    return optimal
end  # wrap
