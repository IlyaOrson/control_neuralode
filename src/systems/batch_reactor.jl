function batch_reactor(; store_results=false::Bool)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    function system!(du, u, p, t, controller)
        # fixed parameters
        α, β, γ, δ = 0.5f0, 1.0f0, 1.0f0, 1.0f0

        # neural network outputs controls taken by the system
        y1, y2 = u
        c1, c2 = controller(u, p)

        # dynamics of the controlled system
        y1_prime = -(c1 + α * c1^2) * y1 + δ * c2
        y2_prime = (β * c1 - γ * c2) * y1

        # update in-place
        @inbounds begin
            du[1] = y1_prime
            du[2] = y2_prime
        end
    end

    # define objective function to optimize
    function loss(params, prob, tsteps)
        sol = solve(prob, INTEGRATOR; p=params, saveat=tsteps, sensealg=SENSEALG)
        return -Array(sol)[2, end]  # second variable, last value, maximize
    end

    # initial conditions and timepoints
    u0 = [1.0f0, 0.0f0]
    tspan = (0.0f0, 1.0f0)
    tsteps = 0.0f0:0.01f0:1.0f0

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(2, 12, tanh_fast),
        FastDense(12, 12, tanh_fast),
        FastDense(12, 2),
        (x, p) -> 5 * sigmoid_fast.(x),  # controllers ∈ (0, 5)
    )

    # current model weights are destructured into a vector of parameters
    θ = initial_params(controller)

    # set differential equation problem
    dudt!(du, u, p, t) = system!(du, u, p, t, controller)
    prob = ODEProblem(dudt!, u0, tspan, θ)

    # variables for streamplots
    phase_time = 0.0f0
    xlims, ylims = (0.0f0, 1.5f0), (0.0f0, 1.5f0)
    coord_lims = [xlims, ylims]
    # xwidth = xlims[end] - xlims[1]
    # ywidth = ylims[end] - ylims[1]
    # start_points_x = range(u0[1] - 1e-4, u0[1] - xwidth/5; length=3)
    # start_points_y = range(u0[2] + 1e-4, u0[2] + ywidth/5; length=3)

    # bug in streamplot won't plot points on the right and upper edges, so a bump is needed
    # https://github.com/matplotlib/matplotlib/issues/21649

    _, states_raw, _ = run_simulation(controller, prob, θ, tsteps)
    marker = FinalState(; points=states_raw[:, end], fmt="m*", markersize=20)
    phase_portrait(
        system!,
        controller,
        θ,
        phase_time,
        coord_lims;
        markers=[marker],
        # start_points_x, start_points_y,
        start_points=reshape(u0 .+ (-1e-4, 0), 1, 2),
        title="Initial policy",
    )

    # closures to comply with required interface
    loss(params) = loss(params, prob, tsteps)
    function plotting_callback(params, loss)
        return plot_simulation(controller, prob, params, tsteps; only=:controls, show=loss)
    end

    @info "Training..."
    result = sciml_train(
        loss,
        θ,
        LBFGS(; linesearch=BackTracking(; iterations=10));
        # iterations=100,
        allow_f_increases=true,
        # cb=plotting_callback,
    )

    store_simulation(
        "optimized",
        controller,
        prob,
        result.minimizer,
        tsteps;
        metadata=Dict(:loss => loss(result.minimizer)),
        datadir,
    )
    plot_simulation(controller, prob, result.minimizer, tsteps; only=:controls)

    θ_opt = result.minimizer

    _, states_opt, _ = run_simulation(controller, prob, θ_opt, tsteps)
    marker = FinalState(; points=states_opt[:, end], fmt="m*", markersize=20)
    return phase_portrait(
        system!,
        controller,
        θ_opt,
        phase_time,
        coord_lims;
        markers=[marker],
        # start_points_x, start_points_y,
        start_points=reshape(u0 .+ (-1e-4, 0), 1, 2),
        title="Optimized policy",
    )
end  # script wrapper
