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
    result = GalacticOptim.solve(optprob, LBFGS(; linesearch))#; cb=plotting_callback)

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

    xlims, ylims = (0f0,1f0), (0f0,1f0)
    xpoints, ypoints = range(xlims...; length=100), range(ylims...; length=100)

    function stream_interface(coords...)
        u = zeros(Float32, 2)
        du = zeros(Float32, 2)
        copyto!(u, coords)
        # du = deepcopy(coords)
        system!(du, u, θ_opt, 0f0, controller)
        return du
    end

    phase_array_tuples = stream_interface.(xpoints', ypoints)
    xphase = getindex.(phase_array_tuples, 1)
    yphase = getindex.(phase_array_tuples, 2)
    magnitude = map((x) -> sum(x.^2), phase_array_tuples)

    # controlling the starting points of the streamlines
    seed_points = hcat(range(xlims...; length=5), range(ylims...; length=5))

    fig = plt.figure()
    # gs = GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    ax = fig.add_subplot()
    strm = ax.streamplot(
        xpoints, ypoints, xphase, yphase,
        color=magnitude, linewidth=2, cmap="autumn", start_points=seed_points
    )
    fig.colorbar(strm.lines)
    ax.set_title("Controlling Starting Points")

    # displaying the starting points
    ax.plot(u0[0], u0[1], "ro")
    ax.plot(seed_points[0], seed_points[1], "bo")
    ax.set(xlim=(-w, w), ylim=(-w, w))

    plt.tight_layout()
    plt.show()

    # δu = (-.2f0, .2f0)
    # N = 10
    # M = 10
    # initial_conditions_variations(
    #     loss, controller, prob, θ_opt, tsteps, datadir, u0, δu, N, M
    # )
end  # script wrapper
