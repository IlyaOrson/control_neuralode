# Solution of a Class of Multistage Dynamic Optimization Problems.
# 2.Problems with Path Constraints

function van_der_pol(; store_results=false::Bool)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    # initial conditions and timepoints
    t0 = 0.0f0
    tf = 5.0f0
    u0 = [0.0f0, 1.0f0, 0.0f0]
    tspan = (t0, tf)
    dt = 0.1f0
    tsteps = t0:dt:tf

    function system!(du, u, p, t, controller, input=:state)
        @argcheck input in (:state, :time)

        # neural network outputs the controls taken by the system
        x1, x2, x3 = u

        if input == :state
            c1 = controller(u, p)[1]  # control based on state and parameters
        elseif input == :time
            c1 = controller(t, p)[1]  # control based on time and parameters
        end

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

    function collocation(
        u0;
        num_supports::Integer=length(tsteps),
        nodes_per_element::Integer=4,
        state_constraints::Bool=false,
    )
        optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
        model = InfiniteModel(optimizer)
        method = OrthogonalCollocation(nodes_per_element)
        @infinite_parameter(
            model, t in [t0, tf], num_supports = num_supports, derivative_method = method
        )

        @variables(
            model,
            begin  # "start" sets the initial guess values
                # state variables
                x[1:3], Infinite(t)
                # control variables
                c[1], Infinite(t)
            end
        )

        # initial conditions
        @constraint(model, [i = 1:3], x[i](0) == u0[i])

        # control range
        @constraint(model, -.3 <= c[1] <= 1.0)

        if state_constraints
            @constraint(model, -.4 <= x[1])
        end

        # dynamic equations
        @constraints(
            model,
            begin
                ∂(x[1], t) == (1 - x[2]^2) * x[1] - x[2] + c[1]
                ∂(x[2], t) == x[1]
                ∂(x[3], t) == x[1]^2 + x[2]^2 + c[1]^2
            end
        )

        @objective(model, Min, x[3](tf))

        optimize!(model)

        model |> optimizer_model |> solution_summary
        states = hcat(value.(x)...) |> permutedims
        controls = hcat(value.(c)...) |> permutedims
        return model, states, controls
    end

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(3, 16, tanh),
        FastDense(16, 16, tanh),
        FastDense(16, 1),
        (x, p) -> (1.3f0 .* σ.(x)) .- 0.3f0,
    )

    # model weights are destructured into a vector of parameters
    θ = initial_params(controller)

    # set differential equation problem
    dudt!(du, u, p, t) = system!(du, u, p, t, controller)
    prob = ODEProblem(dudt!, u0, tspan, θ)

    phase_time = 0.0f0
    function square_bounds(u0, arista)
        low_bounds = u0 .- repeat([arista/2], length(u0))
        high_bounds = u0 .+ repeat([arista/2], length(u0))
        bounds = [(l, h) for (l, h) in zip(low_bounds, high_bounds)]
    end

    _, states_raw, _ = run_simulation(controller, prob, θ, tsteps)

    start_mark = InitialState(; points=states_raw[:, 1])
    marker_path = IntegrationPath(; points=states_raw)
    final_mark = FinalState(; points=states_raw[:, end])
    phase_portrait(
        system!,
        controller,
        θ,
        phase_time,
        square_bounds(u0, 7);
        dimension=3,
        projection=[1, 2],
        markers=[marker_path, start_mark, final_mark],
        # start_points_x, start_points_y,
        # start_points=reshape(u0 .+ repeat([-1e-4], 3), 1, 3),
        title="Initial policy",
    )

    infopt_model, states_collocation, controls_collocation = collocation(u0)
    interpol = interpolant(tsteps, controls_collocation)

    plot_collocation(controls_collocation[1,:], interpol, tsteps)

    # preconditioning to control sequences
    function precondition(t, p)
        Zygote.ignore() do
            return [interpol(t)]
        end
    end
    @info "Collocation result"
    display(lineplot(x -> precondition(x, nothing)[1], t0, tf; xlim=(t0, tf)))
    # display(lineplot(x -> precondition(x, nothing)[2], t0, tf, xlim=(t0,tf)))

    @info "Preconditioning..."
    θ = preconditioner(
        controller,
        precondition,
        system!,
        t0,
        u0,
        tsteps[2:2:end],
        #control_range_scaling=[maximum(controls_collocation) - minimum(controls_collocation)],
    )

    # prob = ODEProblem(dudt!, u0, tspan, θ)
    prob = remake(prob; p=θ)

    plot_simulation(controller, prob, θ, tsteps; only=:controls)
    store_simulation("precondition", controller, prob, θ, tsteps; datadir)

    _, states_raw, _ = run_simulation(controller, prob, θ, tsteps)
    start_mark = InitialState(; points=states_raw[:, 1])
    marker_path = IntegrationPath(; points=states_raw)
    final_mark = FinalState(; points=states_raw[:, end])
    phase_portrait(
        system!,
        controller,
        θ,
        phase_time,
        bounds;
        dimension=3,
        projection=[1, 2],
        markers=[marker_path, start_mark, final_mark],
        # start_points_x, start_points_y,
        # start_points=reshape(u0 .+ repeat([-1e-4], 3), 1, 3),
        title="Preconditioned policy",
    )

    # closures to comply with required interface
    function plotting_callback(params, loss)
        return plot_simulation(
            controller, prob, params, tsteps; only=:states, vars=[1], show=loss
        )
    end

    ### define objective function to optimize
    function loss(params, prob, tsteps)
        # integrate ODE system (stiff problem)
        sol = OrdinaryDiffEq.solve(prob, AutoTsit5(Rosenbrock23()); p=params, saveat=tsteps)
        return Array(sol)[3, end]  # return last value of third variable ...to be minimized
    end
    loss(params) = loss(params, prob, tsteps)

    @info "Training..."
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

    ### now add state constraint x1(t) > -0.4 with
    function penalty_loss(params, prob, tsteps; α=10.0f0)
        # integrate ODE system
        sensealg = InterpolatingAdjoint(;
            autojacvec=ReverseDiffVJP(true), checkpointing=true
        )
        sol = Array(
            OrdinaryDiffEq.solve(
                prob, AutoTsit5(Rosenbrock23()); p=params, saveat=tsteps, sensealg
            ),
        )
        fault = min.(sol[1, 1:end] .+ 0.4f0, 0.0f0)
        penalty = α * dt * sum(fault .^ 2)  # quadratic penalty
        return sol[3, end] + penalty
    end

    penalty_coefficients = [10.0f0, 10f1, 10f2, 10f3]
    for α in penalty_coefficients

        # set differential equation struct again
        constrained_prob = ODEProblem(dudt!, u0, tspan, result.minimizer)
        # function plotting_callback(params, loss)
        #     return plot_simulation(
        #         controller, constrained_prob, params, tsteps; only=:states, vars=[1], show=loss
        #     )
        # end

        # closures to comply with interface
        penalty_loss_(params) = penalty_loss(params, constrained_prob, tsteps; α)

        @info α
        # plot_simulation(
        #     controller,
        #     constrained_prob,
        #     result.minimizer,
        #     tsteps;
        #     only=:controls,
        #     show=penalty_loss_(result.minimizer),
        # )

        adtype = GalacticOptim.AutoZygote()
        optf = OptimizationFunction((x, p) -> penalty_loss_(x), adtype)
        optfunc = GalacticOptim.instantiate_function(
            optf, result.minimizer, adtype, nothing
        )
        optprob = OptimizationProblem(optfunc, result.minimizer; allow_f_increases=true)
        linesearch = BackTracking(; iterations=20)
        result = GalacticOptim.solve(
            optprob,
            LBFGS(; linesearch);
            iterations=50,  # FIXME
            # cb=plotting_callback,
        )
    end
    θ_opt = result.minimizer
    optimal = result.minimum

    constrained_prob = ODEProblem(dudt!, u0, tspan, θ_opt)

    # penalty_loss(result.minimizer, constrained_prob, tsteps; α=penalty_coefficients[end])
    plot_simulation(controller, constrained_prob, θ_opt, tsteps; only=:controls)

    store_simulation(
        "constrained",
        controller,
        constrained_prob,
        θ_opt,
        tsteps;
        datadir,
        metadata=Dict(
            :loss =>
                penalty_loss(θ_opt, constrained_prob, tsteps; α=penalty_coefficients[end]),
            :constraint => "quadratic x2(t) > -0.4",
        ),
    )

    _, states_opt, _ = run_simulation(controller, prob, θ_opt, tsteps)
    start_mark = InitialState(; points=states_opt[:, 1])
    marker_path = IntegrationPath(; points=states_opt)
    final_mark = FinalState(; points=states_opt[:, end])
    shader = ShadeConf(; indicator=function (x, y)
        if x > -.4
            return true
        end
        return false
    end)
    phase_portrait(
        system!,
        controller,
        θ_opt,
        phase_time,
        bounds;
        shader,
        dimension=3,
        projection=[1, 2],
        markers=[marker_path, start_mark, final_mark],
        # start_points_x, start_points_y,
        # start_points=reshape(u0 .+ repeat([-1e-4], 3), 1, 3),
        title="Optimized policy",
    )

    # u0 = [0f0, 1f0, 0f0]
    perturbation_specs = [
        (variable=1, type=:positive, scale=1.0f0, samples=8, percentage=2f-2)
        (variable=2, type=:negative, scale=1.0f0, samples=8, percentage=2f-2)
        (variable=3, type=:positive, scale=20.0f0, samples=8, percentage=2f-2)
    ]
    constraint_spec = ConstRef(; val=-.4, direction=:horizontal, class=:state, var=1)
    # plot_initial_perturbations(
    #     controller, prob, θ_opt, tsteps, u0, perturbation_specs; refs=[constraint_spec]
    # )

    plot_initial_perturbations_collocation(
        controller,
        prob,
        θ_opt,
        tsteps,
        u0,
        perturbation_specs,
        collocation;
        refs=[constraint_spec],
        storedir=generate_data_subdir(@__FILE__),
    )
    return optimal
end  # wrapper script
