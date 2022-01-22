# Hicks, G. A., & Ray, W. H. (1971).
# Approximation methods for optimal control synthesis.
# The Canadian Journal of Chemical Engineering, 49(4), 522-528.

function reference_tracking(; store_results=false::Bool)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    # irreversible reaction
    reaction = "irreversible"
    time = 20.0f0  # final time (θ in original paper)
    cf = 1.0f0
    Tf = 300.0f0
    Tc = 290.0f0
    J = 100.0f0
    α = 1.95f-4
    k10 = 300.0f0
    N = 25.2f0
    u_upper = 1500.0f0
    u_lower = 0.0f0
    y1s = 0.408126f0
    y2s = 3.29763f0
    us = 370.0f0

    # # reversible reaction
    # reaction = "reversible"
    # time = 1f0  # final time (θ in original paper)
    # cf = 1f0
    # Tf = 323f0
    # Tc = 326f0
    # J = 100f0
    # α = 1f0
    # k10 = 1.5f7
    # k20 = 1.5f10
    # N = 10f0
    # γ = 1.5f0
    # u_upper = 9.5f0
    # u_lower = 0f0
    # y1s = 0.433848f0
    # y2s = 0.659684f0
    # us = 3.234f0

    # adimensional constants
    yf = Tf / (J * cf)
    yc = Tc / (J * cf)

    # case 1
    # α1 = 1f6
    # α2 = 2f3
    # α3 = 1f-3

    # case 2
    # α1 = 1f6
    # α2 = 2f3
    # α3 = 0f0

    # case 3
    # α1 = 10f0
    # α2 = 1f0
    # α3 = 0.1f0

    # case 4
    # α1 = 10f0
    # α2 = 1f0
    # α3 = 0f0

    # custom case
    α1 = 1.0f0
    α2 = 1.0f1
    α3 = 1.0f-1

    # initial conditions and timepoints
    @show u0 = [1.0f0, yf]
    @show tspan = (0.0f0, time)
    Δt = 1f-2

    function system!(du, u, p, t, controller; input=:state)
        @argcheck input in (:state, :time)

        y1, y2 = u

        if input == :state
            c = controller(u, p)[1]
        elseif input == :time
            c = controller(t, p)[1]
        end

        # reaction rate
        r = k10 * y1 * exp(-N / y2)  # irreversible (case 1 & case 2)
        # r = k10 * y1 * exp(-N/y2) - k20 * exp(-γ*N/y2) * (1-y1)  # reversible (case 3 & case 4)

        # dynamics of the controlled system
        y1_prime = (1 - y1) / time - r
        y2_prime = (yf - y2) / time + r - α * c * (y2 - yc)

        # update in-place
        @inbounds begin
            du[1] = y1_prime
            du[2] = y2_prime
        end
    end

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(2, 16, tanh_fast),
        FastDense(16, 16, tanh_fast),
        FastDense(16, 2),
        (x, p) -> [u_lower + (u_upper - u_lower) * sigmoid_fast(x[1])],  # controllers ∈ [u_lower, u_upper]
    )

    controlODE = ControlODE(controller, system!, u0, tspan; Δt)

    # model weights are destructured into a vector of parameters
    θ = initial_params(controller)

    # define objective function to optimize
    function loss(controlODE, params)

        # curious error with ROS3P()
        sol = solve(controlODE, params) |> Array # integrate ODE system

        sum_squares = 0.0f0
        for state in eachcol(sol)
            control = controller(state, params)
            sum_squares +=
                α1 * (state[1] - y1s)^2 + α2 * (state[2] - y2s)^2 + α3 * (control[1] - us)^2
        end
        return sum_squares * 0.01f0
    end

    # closures to comply with required interface
    loss(params) = loss(controlODE, params)
    function plotting_callback(params, loss)
        return plot_simulation(controlODE, params; only=:controls, show=loss)
    end

    plot_simulation(controlODE, θ; only=:controls, show=loss(θ))

    result = sciml_train(
        loss,
        θ,
        LBFGS(; linesearch=BackTracking());
        allow_f_increases=true,
        cb=plotting_callback,
    )

    plot_simulation(
        controlODE,
        result.minimizer;
        only=:states,
        vars=[1],
        show=loss(result.minimizer),
        yrefs=[y1s],
    )
    plot_simulation(
        controlODE,
        result.minimizer;
        only=:states,
        vars=[2],
        show=loss(result.minimizer),
        yrefs=[y2s],
    )
    plot_simulation(
        controlODE,
        result.minimizer;
        only=:controls,
        show=loss(result.minimizer),
        yrefs=[us],
    )

    metadata = Dict(
        :loss => loss(result.minimizer),
        :reaction => reaction,
        :α1 => α1,
        :α2 => α2,
        :α3 => α3,
    )
    return store_simulation(
        reaction, controlODE, result.minimizer; metadata, datadir
    )
end  # script wrapper
