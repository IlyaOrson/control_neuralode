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

    system = BatchReactor()
    controlODE = ControlODE(system)

    θ = initial_params(controlODE.controller)

    # variables for streamplots
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
        # start_points=reshape(u0 .+ (-1e-4, 0), 1, 2),
        title="Initial policy",
    )

    # closures to comply with required interface
    loss(params) = loss(controlODE, params)
    function plotting_callback(params, loss)
        return plot_simulation(controlODE, params; only=:controls, show=loss)
    end

    @info "Training..."
    grad!(g, params) = g .= Zygote.gradient(loss, params)[1]
    θ = optimize_optim(θ, loss, grad!)

    store_simulation(
        "optimized",
        controlODE,
        θ;
        datadir,
    )
    plot_simulation(controlODE, θ; only=:controls)

    _, states_raw, _ = run_simulation(controlODE, θ)
    phase_portrait(
        controlODE,
        θ,
        coord_lims;
        markers=states_markers(states_raw),
        # start_points_x, start_points_y,
        # start_points=reshape(u0 .+ (-1e-4, 0), 1, 2),
        title="Optimized policy",
    )
    return
end  # script wrapper
