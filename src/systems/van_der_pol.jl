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
    Δt = 0.1f0

    function system!(du, u, p, t, controller; input=:state)
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

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(3, 16, tanh_fast),
        FastDense(16, 16, tanh_fast),
        FastDense(16, 1),
        (x, p) -> (1.3f0 .* sigmoid_fast.(x)) .- 0.3f0,
    )

    controlODE = ControlODE(controller, system!, u0, tspan; Δt)

    θ = initial_params(controlODE.controller)

    function collocation(
        u0;
        num_supports::Integer=length(controlODE.tsteps),
        nodes_per_element::Integer=4,
        constrain_states::Bool=false,
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
        @constraint(model, -0.3 <= c[1] <= 1.0)

        if constrain_states
            @constraint(model, -0.4 <= x[1])
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

        return model, supports(t), states, controls
    end

    phase_time = 0.0f0
    function square_bounds(u0, arista)
        low_bounds = u0 .- repeat([arista / 2], length(u0))
        high_bounds = u0 .+ repeat([arista / 2], length(u0))
        bounds = [(l, h) for (l, h) in zip(low_bounds, high_bounds)]
        return bounds
    end

    _, states_raw, _ = run_simulation(controlODE, θ)

    start_mark = InitialState(; points=states_raw[:, 1])
    marker_path = IntegrationPath(; points=states_raw)
    final_mark = FinalState(; points=states_raw[:, end])
    phase_portrait(
        controlODE,
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

    control_profile, infopt_model, times_collocation, states_collocation, controls_collocation = collocation_preconditioner(
        u0, collocation; plot=true
    )

    θ = preconditioner(
        controlODE,
        control_profile;
        #control_range_scaling=[maximum(controls_collocation) - minimum(controls_collocation)],
    )

    plot_simulation(controlODE, θ; only=:controls)
    store_simulation("precondition", controlODE, θ; datadir)

    _, states_raw, _ = run_simulation(controlODE, θ)
    start_mark = InitialState(; points=states_raw[:, 1])
    marker_path = IntegrationPath(; points=states_raw)
    final_mark = FinalState(; points=states_raw[:, end])
    phase_portrait(
        controlODE,
        θ,
        phase_time,
        square_bounds(u0, 7);
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
            controlODE, params; only=:states, vars=[1], show=loss
        )
    end

    ### define objective function to optimize
    function loss(controlODE, params; kwargs...)
        sol = solve(controlODE, params; kwargs...)
        return Array(sol)[3, end]  # return last value of third variable ...to be minimized
    end
    loss(params) = loss(controlODE, params)

    @info "Training..."
    @infiltrate
    result = sciml_train(
        loss,
        θ,
        LBFGS(; linesearch=BackTracking());
        # cb=plotting_callback,
        allow_f_increases=true,
    )
    θ = result.minimizer

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
        sol = Array(solve(controlODE, params; kwargs...))
        fault = min.(sol[1, 1:end] .+ 0.4f0, 0.0f0)
        penalty = α * Δt * sum(fault .^ 2)  # quadratic penalty
        return sol[3, end] + penalty
    end
    @info "Enforcing constraints..."
    penalty_coefficients = [10.0f0, 10.0f1, 10.0f2, 10.0f3]
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

        result = sciml_train(
            penalty_loss_,
            θ,
            LBFGS(; linesearch=BackTracking(; iterations=20));
            iterations=50,
            allow_f_increases=true,
            # cb=plotting_callback,
        )
        θ = result.minimizer
        next!(prog; showvalues=[(:α, α), (:loss, penalty_loss_(θ))])
    end
    θ_opt = result.minimizer
    optimal = result.minimum

    # penalty_loss(result.minimizer, constrained_prob, tsteps; α=penalty_coefficients[end])
    plot_simulation(controlODE, θ_opt; only=:controls)

    store_simulation(
        "constrained",
        controlODE,
        θ_opt;
        datadir,
        metadata=Dict(
            :loss =>
                penalty_loss(controlODE, θ_opt; α=penalty_coefficients[end]),
            :constraint => "quadratic x2(t) > -0.4",
        ),
    )

    _, states_opt, _ = run_simulation(controlODE, θ_opt)
    start_mark = InitialState(; points=states_opt[:, 1])
    marker_path = IntegrationPath(; points=states_opt)
    final_mark = FinalState(; points=states_opt[:, end])
    shader = ShadeConf(; indicator=function (x, y)
        if x > -0.4
            return true
        end
        return false
    end)
    phase_portrait(
        controlODE,
        θ_opt,
        phase_time,
        square_bounds(u0, 7);
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
        (variable=1, type=:positive, scale=1.0f0, samples=8, percentage=2.0f-2)
        (variable=2, type=:negative, scale=1.0f0, samples=8, percentage=2.0f-2)
        (variable=3, type=:positive, scale=20.0f0, samples=8, percentage=2.0f-2)
    ]
    constraint_spec = ConstRef(; val=-0.4, direction=:horizontal, class=:state, var=1)

    plot_initial_perturbations_collocation(
        controlODE,
        θ_opt,
        perturbation_specs,
        collocation;
        refs=[constraint_spec],
        storedir=generate_data_subdir(@__FILE__),
    )
    return optimal
end  # wrap
