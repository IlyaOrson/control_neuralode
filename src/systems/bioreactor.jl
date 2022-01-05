# Bradford, E., Imsland, L., Zhang, D., & del Rio Chanona, E. A. (2020).
# Stochastic data-driven model predictive control using Gaussian processes.
# Computers & Chemical Engineering, 139, 106844.

# objective: maximize C_qc

# state constraints
# C_N(t) - 150 ≤ 0              t = T
# C_N(t) − 800 ≤ 0              ∀t
# 0.011 C_X(t) - C_qc(t) ≤ 3f-2 ∀t

function bioreactor(; store_results=false::Bool)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    # initial conditions and timepoints
    t0 = 0.0f0
    tf = 240.0f0
    Δt = 10.0f0
    C_X₀, C_N₀, C_qc₀ = 1.0f0, 150.0f0, 0.0f0
    u0 = [C_X₀, C_N₀, C_qc₀]
    tspan = (t0, tf)
    tsteps = t0:Δt:tf
    control_ranges = [(120.0f0, 400.0f0), (0.0f0, 40.0f0)]

    system_params = (
        u_m=0.0572f0,
        u_d=0.0f0,
        K_N=393.1f0,
        Y_NX=504.5f0,
        k_m=0.00016f0,
        k_d=0.281f0,
        k_s=178.9f0,
        k_i=447.1f0,
        k_sq=23.51f0,
        k_iq=800.0f0,
        K_Np=16.89f0,
    )

    function system!(du, u, p, t, controller, input=:state)
        @argcheck input in (:state, :time)

        u_m, u_d, K_N, Y_NX, k_m, k_d, k_s, k_i, k_sq, k_iq, K_Np = values(system_params)

        # neural network outputs controls based on state
        C_X, C_N, C_qc = u
        if input == :state
            I, F_N = controller(u, p)
        elseif input == :time
            I, F_N = controller(t, p)
        end

        # auxiliary variables
        I_ksi = I / (I + k_s + I^2.0f0 / k_i)
        CN_KN = C_N / (C_N + K_N)

        I_kiq = I / (I + k_sq + I^2.0f0 / k_iq)
        Cqc_KNp = C_qc / (C_N + K_Np)

        # dynamics of the controlled system
        dC_X = u_m * I_ksi * C_X * CN_KN - u_d * C_X
        dC_N = -Y_NX * u_m * I_ksi * C_X * CN_KN + F_N
        dC_qc = k_m * I_kiq * C_X - k_d * Cqc_KNp

        # update in-place
        @inbounds begin
            du[1] = dC_X
            du[2] = dC_N
            du[3] = dC_qc
        end
    end

    function collocation(
        u0;
        num_supports::Integer=length(tsteps),
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
            begin
                # state variables
                x[1:3], Infinite(t)
                # control variables
                c[1:2], Infinite(t)
            end
        )

        # fixed parameters
        u_m, u_d, K_N, Y_NX, k_m, k_d, k_s, k_i, k_sq, k_iq, K_Np = values(system_params)

        # initial conditions
        @constraint(model, [i = 1:3], x[i](0) == u0[i])

        # control range
        @constraints(
            model,
            begin
                control_ranges[1][1] <= c[1] <= control_ranges[1][2]
                control_ranges[2][1] <= c[2] <= control_ranges[2][2]
            end
        )

        if constrain_states
            @constraints(
                model,
                begin
                    x[2] <= 150
                    x[2] <= 800
                    1.1f-2 * x[1] - x[3] <= 3f-2
                end
            )
        end

        # dynamic equations
        @constraints(
            model,
            begin
                ∂(x[1], t) ==
                u_m *
                (c[1] / (c[1] + k_s + c[1]^2.0f0 / k_i)) *
                x[1] *
                (x[2] / (x[2] + K_N)) - u_d * x[1]
                ∂(x[2], t) ==
                -Y_NX *
                u_m *
                (c[1] / (c[1] + k_s + c[1]^2.0f0 / k_i)) *
                x[1] *
                (x[2] / (x[2] + K_N)) + c[2]
                ∂(x[3], t) ==
                k_m * (c[1] / (c[1] + k_sq + c[1]^2.0f0 / k_iq)) * x[1] -
                k_d * (x[3] / (x[2] + K_Np))
            end
        )

        @objective(model, Max, x[3](tf))

        optimize!(model)

        model |> optimizer_model |> solution_summary
        states = hcat(value.(x)...) |> permutedims
        controls = hcat(value.(c)...) |> permutedims
        return model, supports(t), states, controls
    end

    # set arquitecture of neural network controller
    controller = FastChain(
        (x, p) -> [x[1], x[2] / 10.0f0, x[3] * 10.0f0],  # input scaling
        FastDense(3, 16, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        FastDense(16, 16, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        FastDense(16, 2; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        # I ∈ [120, 400] & F ∈ [0, 40] in Bradford 2020
        # (x, p) -> [280f0 * sigmoid(x[1]) + 120f0, 40f0 * sigmoid(x[2])],
        scaled_sigmoids(control_ranges),
    )

    # initial parameters
    controller_shape(controller)
    θ = initial_params(controller)  # destructure model weights into a vector of parameters

    # set differential equation problem
    dudt!(du, u, p, t) = system!(du, u, p, t, controller)
    prob = ODEProblem(dudt!, u0, tspan, θ)

    control_profile, infopt_model, times_collocation, states_collocation, controls_collocation = collocation_preconditioner(u0, collocation; plot=true)

    θ = preconditioner(
        controller,
        control_profile,
        system!,
        t0,
        u0,
        tsteps[2:2:end];
        progressbar=true,
        plot_progress=false,
        # control_range_scaling=[range[end] - range[1] for range in control_ranges],
    )

    # prob = ODEProblem(dudt!, u0, tspan, θ)
    prob = remake(prob; p=θ)

    plot_simulation(controller, prob, θ, tsteps; only=:states)
    plot_simulation(controller, prob, θ, tsteps; only=:controls)
    store_simulation("precondition", controller, prob, θ, tsteps; datadir)

    function state_penalty_functional(
        solution_array, time_intervals; state_penalty=relaxed_log_barrier, δ=1f1
    )
        @assert size(solution_array, 2) == length(time_intervals) + 1

        ratio_X_N = 3f-2 / 800.0f0
        C_N_over = map(y -> state_penalty(800.0f0 - y; δ), solution_array[2, 1:end])
        C_X_over = map(
            (x, z) -> state_penalty(3f-2 - (1.1f-2 * x - z); δ=δ * ratio_X_N),
            solution_array[1, 1:end],
            solution_array[3, 1:end],
        )
        C_N_over_last = state_penalty(150.0f0 - solution_array[2, end]; δ)

        return sum((C_N_over .+ C_X_over)[1:(end - 1)] .* time_intervals) + C_N_over_last
    end

    # state constraints on control change
    # C_N(t) - 150 ≤ 0              t = T
    # C_N(t) − 800 ≤ 0              ∀t
    # 0.011 C_X(t) - C_qc(t) ≤ 3f-2 ∀t
    function loss(
        params,
        prob;
        state_penalty=relaxed_log_barrier,
        δ=1f1,
        α=1f-3,
        μ=(3.125f-8, 3.125f-6),
        ρ=1f-1,
        tsteps=(),
    )

        # integrate ODE system
        sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true)
        sol_raw = solve(
            prob, BS3(); p=params, saveat=tsteps, abstol=1f-1, reltol=1f-1, sensealg
        )
        sol = Array(sol_raw)

        # approximate integral penalty
        # state_penalty = Δt * (sum(C_N_over) + sum(C_X_over)) + C_N_over_last  # for fixed timesteps
        Zygote.ignore() do
            global time_intervals = [
                sol_raw.t[i + 1] - sol_raw.t[i] for i in eachindex(sol_raw.t[1:(end - 1)])
            ]
        end

        state_penalty = α * state_penalty_functional(sol, time_intervals; δ, state_penalty)

        # penalty on change of controls
        control_penalty = 0.0f0
        for i in 1:(size(sol, 2) - 1)
            prev = controller(sol[:, i], params)
            post = controller(sol[:, i + 1], params)
            for j in 1:length(prev)
                control_penalty += μ[j] * (prev[j] - post[j])^2
            end
        end

        regularization = ρ * mean(abs2, θ)  # sum(abs2, θ)

        objective = -sol[3, end]  # maximize C_qc

        return objective, state_penalty, control_penalty, regularization
    end

    function plots_callback(controller, prob, θ, tsteps)
        plot_simulation(
            controller, prob, θ, tsteps; only=:states, vars=[2], yrefs=[800, 150]
        )
        plot_simulation(
            controller,
            prob,
            θ,
            tsteps;
            only=:states,
            fun=(x, y, z) -> 1.1f-2x - z,
            yrefs=[3f-2],
        )
        return plot_simulation(controller, prob, θ, tsteps; only=:controls)
    end

    # α: penalty coefficient
    # δ: barrier relaxation coefficient
    α0, δ0 = 1f-5, 100.0f0
    barrier_iterations = 0:20
    αs = [α0 for _ in barrier_iterations]
    δs = [δ0 * 0.8f0^i for i in barrier_iterations]
    θ = constrained_training(
        controller,
        prob,
        θ,
        loss;
        αs,
        δs,
        tsteps,
        show_progressbar=true,
        # plots_callback,
        datadir,
    )

    final_values = NamedTuple{(
        :objective, :state_penalty, :control_penalty, :regularization
    )}(
        loss(θ, prob; δ=δs[end], α=αs[end], tsteps)
    )

    @info "Final states"
    # plot_simulation(controller, prob, θ, tsteps; only=:states, vars=[1], show=final_values)
    plot_simulation(
        controller,
        prob,
        θ,
        tsteps;
        only=:states,
        vars=[2],
        show=final_values,
        yrefs=[800, 150],
    )
    plot_simulation(controller, prob, θ, tsteps; only=:states, vars=[3], show=final_values)
    plot_simulation(
        controller,
        prob,
        θ,
        tsteps;
        only=:states,
        fun=(x, y, z) -> 1.1f-2x - z,
        yrefs=[3f-2],
    )

    @info "Final controls"
    plot_simulation(
        controller, prob, θ, tsteps; only=:controls, vars=[1], show=final_values
    )
    plot_simulation(
        controller, prob, θ, tsteps; only=:controls, vars=[2], show=final_values
    )

    # initial conditions and timepoints
    # t0 = 0f0
    # tf = 240f0
    # Δt = 10f0
    # C_X₀, C_N₀, C_qc₀ = 1f0, 150f0, 0f0
    # u0 = [C_X₀, C_N₀, C_qc₀]
    # tspan = (t0, tf)
    # tsteps = t0:Δt:tf
    # control_ranges = [(120f0, 400f0), (0f0, 40f0)]
    perturbation_specs = [
        (variable=1, type=:centered, scale=10.0f0, samples=10, percentage=2f-2)
        (variable=2, type=:centered, scale=800.0f0, samples=10, percentage=2f-2)
        (variable=3, type=:positive, scale=5f-1, samples=10, percentage=2f-2)
    ]
    return plot_initial_perturbations(controller, prob, θ, tsteps, u0, perturbation_specs)
end  # script wrapper
