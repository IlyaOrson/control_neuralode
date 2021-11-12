# Solution of a Class of Multistage Dynamic Optimization Problems.
# 2.Problems with Path Constraints

function van_der_pol(; store_results=true::Bool)

    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    function system!(du, u, p, t, controller)
        # neural network outputs controls taken by the system
        x1, x2, x3 = u
        c1 = controller(u, p)[1]

        # dynamics of the controlled system
        x1_prime = (1 - x2^2) * x1 - x2 + c1
        x2_prime = x1
        x3_prime = x1^2 + x2^2 + c1^2

        # update in-place
        @inbounds begin
            du[1] = x1_prime
            du[2] = x2_prime
            du[3] = x3_prime
        end
    end

    # initial conditions and timepoints
    t0 = 0.0f0
    tf = 5.0f0
    u0 = [0.0f0, 1.0f0, 0.0f0]
    tspan = (t0, tf)
    dt = 0.1f0
    tsteps = t0:dt:tf

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(3, 16, tanh),
        FastDense(16, 16, tanh),
        FastDense(16, 1),
        (x, p) -> (1.3f0 .* σ.(x)) .- 0.3f0,
    )

    # model weights are destructured into a vector of parameters
    θ = initial_params(controller)

    # set differential equation problem and solve it
    dudt!(du, u, p, t) = system!(du, u, p, t, controller)

    # closures to comply with required interface
    prob = ODEProblem(dudt!, u0, tspan, θ)
    function plotting_callback(params, loss)
        return plot_simulation(
            controller, prob, params, tsteps; only=:states, vars=[1], show=loss
        )
    end

    ### define objective function to optimize
    function loss(params, prob, tsteps)
        # integrate ODE system (stiff problem)
        sol = solve(prob, AutoTsit5(Rosenbrock23()); p=params, saveat=tsteps)
        return Array(sol)[3, end]  # return last value of third variable ...to be minimized
    end
    loss(params) = loss(params, prob, tsteps)

    # adtype = GalacticOptim.AutoZygote()
    # optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
    # optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
    # optprob = GalacticOptim.OptimizationProblem(optfunc, θ; allow_f_increases=true)
    # result = GalacticOptim.solve(
    #     optprob, LBFGS(; linesearch=BackTracking()); cb=plotting_callback
    # )

    result = DiffEqFlux.sciml_train(
        loss,
        θ,
        LBFGS(; linesearch=BackTracking());
        # cb=plotting_callback,
        allow_f_increases=true,
        f_tol=1f-1,
    )

    store_simulation(
        "unconstrained",
        controller,
        prob,
        result.minimizer,
        tsteps;
        datadir,
        metadata=Dict(:loss => loss(result.minimizer), :constraint => "none"),
    )

    ### now add state constraint x2(t) > -0.4 with
    function penalty_loss(params, prob, tsteps; α=10.0f0)
        # integrate ODE system (stiff problem)
        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true), checkpointing=true)
        sol = solve(prob, AutoTsit5(Rosenbrock23()); p=params, saveat=tsteps, sensealg) |> Array
        fault = min.(sol[1, 1:end] .+ 0.4f0, 0f0)
        penalty = α * dt * sum(fault .^ 2)  # quadratic penalty
        return sol[3, end] + penalty
    end

    penalty_coefficients = [10.0f0, 10^2.0f0, 10^3.0f0, 10^4.0f0]
    for α in penalty_coefficients
        # global result
        # @show result

        # set differential equation struct again
        constrained_prob = ODEProblem(dudt!, u0, tspan, result.minimizer)
        # function plotting_callback(params, loss)
        #     return plot_simulation(
        #         controller, constrained_prob, params, tsteps; only=:states, vars=[1], show=loss
        #     )
        # end

        # closures to comply with interface
        penalty_loss_(params) = penalty_loss(params, constrained_prob, tsteps; α)

        @info "Control Profile" α
        plot_simulation(
            controller,
            constrained_prob,
            result.minimizer,
            tsteps;
            only=:controls,
            show=penalty_loss_(result.minimizer),
        )

        adtype = AutoZygote()
        optf = OptimizationFunction((x, p) -> penalty_loss_(x), adtype)
        optfunc = GalacticOptim.instantiate_function(
            optf, result.minimizer, adtype, nothing
        )
        optprob = OptimizationProblem(
            optfunc, result.minimizer; allow_f_increases=true
        )
        linesearch = BackTracking(iterations=20)
        result = GalacticOptim.solve(
            optprob, LBFGS(; linesearch)#; cb=plotting_callback
        )
    end

    constrained_prob = ODEProblem(dudt!, u0, tspan, result.minimizer)

    penalty_loss(params, constrained_prob, tsteps; α)
    plot_simulation(
        controller,
        constrained_prob,
        result.minimizer,
        tsteps;
        only=:controls,
    )

    store_simulation(
        "constrained",
        controller,
        constrained_prob,
        result.minimizer,
        tsteps;
        datadir,
        metadata=Dict(
            :loss => penalty_loss(
                result.minimizer, constrained_prob, tsteps; α=penalty_coefficients[end]
            ),
            :constraint => "quadratic x2(t) > -0.4",
        ),
    )
end  # wrapper script
