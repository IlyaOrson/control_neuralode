function batch_reactor(; store_results=true::Bool)

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
        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true), checkpointing=true)
        sol = solve(prob, Tsit5(); p=params, saveat=tsteps, sensealg)  # integrate ODE system
        return -Array(sol)[2, end]  # second variable, last value, maximize
    end

    # initial conditions and timepoints
    u0 = [1.0f0, 0.0f0]
    tspan = (0.0f0, 1.0f0)
    tsteps = 0.0f0:0.01f0:1.0f0

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(2, 12, tanh),
        FastDense(12, 12, tanh),
        FastDense(12, 2),
        (x, p) -> 5 * σ.(x),  # controllers ∈ (0, 5)
    )

    # current model weights are destructured into a vector of parameters
    θ = initial_params(controller)

    # set differential equation problem and solve it
    dudt!(du, u, p, t) = system!(du, u, p, t, controller)
    prob = ODEProblem(dudt!, u0, tspan, θ)

    # closures to comply with required interface
    loss(params) = loss(params, prob, tsteps)
    function plotting_callback(params, loss)
        return plot_simulation(controller, prob, params, tsteps; only=:controls, show=loss)
    end

    @info "Optimizing"
    adtype = GalacticOptim.AutoZygote()
    optf = OptimizationFunction((x, p) -> loss(x), adtype)
    optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
    optprob = OptimizationProblem(optfunc, θ; allow_f_increases=true)
    linesearch = BackTracking(iterations=10)
    result = GalacticOptim.solve(optprob, LBFGS(; linesearch), iterations=20)#; cb=plotting_callback)

    store_simulation(
        "optimized",
        controller,
        prob,
        result.minimizer,
        tsteps;
        metadata=Dict(:loss => loss(result.minimizer)),
        datadir
    )
    plot_simulation(controller, prob, result.minimizer, tsteps; only=:controls)

    θ_opt = result.minimizer

    _, states_opt, _ = run_simulation(controller, prob, θ_opt , tsteps)

    phase_time = 0f0
    xlims, ylims = (0f0,1f0), (0f0,1f0)
    start_points_x = range(u0[1], u0[1] - .2; length=2)
    start_points_y = range(u0[2], u0[2] + .2; length=2)
    phase_plot(
        system!, controller, θ_opt, phase_time, xlims, ylims;
        initial_point=states_opt[:,1], final_point=states_opt[:,end],
        start_points_x, start_points_y,
    )

    # δu = (-.2f0, .2f0)
    # N = 10
    # M = 10
    # initial_conditions_variations(
    #     loss, controller, prob, θ_opt, tsteps, datadir, u0, δu, N, M
    # )
end  # script wrapper
