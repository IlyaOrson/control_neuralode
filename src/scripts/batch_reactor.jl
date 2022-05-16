function batch_reactor(; store_results::Bool=false)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    # define objective function to optimize
    function loss(controlODE, params)
        sol = solve(controlODE, params)
        return -Array(sol)[2, end]  # second variable, last value, maximize
    end

    # initial conditions and timepoints
    u0 = [1.0f0, 0.0f0]
    tspan = (0.0f0, 1.0f0)
    Δt = 1f-2

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(2, 12, tanh_fast),
        FastDense(12, 12, tanh_fast),
        FastDense(12, 2),
        (x, p) -> 5 * sigmoid_fast.(x),  # controllers ∈ (0, 5)
    )

    system = BatchReactor()
    controlODE = ControlODE(controller, system, u0, tspan; Δt)

    θ = initial_params(controller)

    # variables for streamplots
    phase_time = 0.0f0
    xlims, ylims = (0.0f0, 1.5f0), (0.0f0, 0.8f0)
    coord_lims = [xlims, ylims]
    # xwidth = xlims[end] - xlims[1]
    # ywidth = ylims[end] - ylims[1]
    # start_points_x = range(u0[1] - 1e-4, u0[1] - xwidth/5; length=3)
    # start_points_y = range(u0[2] + 1e-4, u0[2] + ywidth/5; length=3)

    _, states_raw, _ = run_simulation(controlODE, θ)
    phase_portrait(
        controlODE,
        θ,
        coord_lims;
        markers=states_markers(states_raw),
        # start_points_x, start_points_y,
        start_points=reshape(u0 .+ (-1e-4, 0), 1, 2),
        title="Initial policy",
    )

    # closures to comply with required interface
    loss(params) = loss(controlODE, params)
    function plotting_callback(params, loss)
        return plot_simulation(controlODE, params; only=:controls, show=loss)
    end

    @info "Training..."
    result = sciml_train(
        loss,
        θ,
        LBFGS(; linesearch=BackTracking());
        # iterations=100,
        # allow_f_increases=true,
        # cb=plotting_callback,
    )

    store_simulation(
        "optimized",
        controlODE,
        result.minimizer;
        metadata=Dict(:loss => loss(result.minimizer)),
        datadir,
    )
    plot_simulation(controlODE, result.minimizer; only=:controls)

    θ = result.minimizer

    _, states_raw, _ = run_simulation(controlODE, θ)
    return phase_portrait(
        controlODE,
        θ,
        coord_lims;
        markers=states_markers(states_raw),
        # start_points_x, start_points_y,
        start_points=reshape(u0 .+ (-1e-4, 0), 1, 2),
        title="Optimized policy",
    )
end  # script wrapper
