function batch_reactor(; store_results=false::Bool)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    function system!(du, u, p, t, controller; input=:state)
        @argcheck input in (:state, :time)

        # fixed parameters
        α, β, γ, δ = 0.5f0, 1.0f0, 1.0f0, 1.0f0

        y1, y2 = u

        # neural network outputs controls taken by the system
        if input == :state
            c1, c2 = controller(u, p)
        elseif input == :time
            c1, c2 = controller(t, p)
        end

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

    controlODE = ControlODE(controller, system!, u0, tspan; Δt)

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
    start_mark = InitialMarkers(; points=states_raw[:, 1])
    marker_path = IntegrationPath(; points=states_raw)
    final_marker = FinalMarkers(; points=states_raw[:, end])
    phase_portrait(
        controlODE,
        θ,
        coord_lims;
        markers=[start_mark, marker_path, final_marker],
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
        LBFGS(; linesearch=BackTracking(; iterations=10));
        # iterations=100,
        allow_f_increases=true,
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
    start_mark = InitialMarkers(; points=states_raw[:, 1])
    marker_path = IntegrationPath(; points=states_raw)
    final_marker = FinalMarkers(; points=states_raw[:, end])
    return phase_portrait(
        controlODE,
        θ,
        coord_lims;
        markers=[start_mark, marker_path, final_marker],
        # start_points_x, start_points_y,
        start_points=reshape(u0 .+ (-1e-4, 0), 1, 2),
        title="Optimized policy",
    )
end  # script wrapper
