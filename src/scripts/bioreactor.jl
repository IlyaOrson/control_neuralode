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
    tspan = (t0, tf)
    C_X₀, C_N₀, C_qc₀ = 1.0f0, 150.0f0, 0.0f0
    u0 = [C_X₀, C_N₀, C_qc₀]
    control_ranges = [(120.0f0, 400.0f0), (0.0f0, 40.0f0)]

    # set arquitecture of neural network controller
    # weights initializer reference https://pytorch.org/docs/stable/nn.init.html
    controller = FastChain(
        (x, p) -> [x[1], x[2] / 100.0f0, x[3] * 10.0f0],  # input scaling
        FastDense(3, 16, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        FastDense(16, 16, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        FastDense(16, 2; initW=(x, y) -> glorot_uniform(x, y)),
        # I ∈ [120, 400] & F ∈ [0, 40] in Bradford 2020
        # (x, p) -> [280f0 * sigmoid(x[1]) + 120f0, 40f0 * sigmoid(x[2])],
        scaled_sigmoids(control_ranges),
    )

    system! = BioReactor()
    controlODE = ControlODE(controller, system!, u0, tspan; Δt)

    function infopt_collocation(;
        u0=controlODE.u0,
        tspan=controlODE.tspan,
        num_supports::Integer=length(controlODE.tsteps),
        nodes_per_element::Integer=4,
        constrain_states::Bool=false,
    )
        optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
        model = InfiniteModel(optimizer)
        method = OrthogonalCollocation(nodes_per_element)
        @infinite_parameter(
            model,
            t in [tspan[1], tspan[2]],
            num_supports = num_supports,
            derivative_method = method
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
        (; u_m, u_d, K_N, Y_NX, k_m, k_d, k_s, k_i, k_sq, k_iq, K_Np) = BioReactor()

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
                    1.1f-2 * x[1] - x[3] <= 3.0f-2
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

        optimize_infopt!(model)

        times = supports(t)
        states = hcat(value.(x)...) |> permutedims
        controls = hcat(value.(c)...) |> permutedims
        return (; model, times, states, controls)
    end

    collocation = infopt_collocation()
    reference_controller = interpolant_controller(collocation; plot=false)

    θ = preconditioner(
        controlODE,
        reference_controller;
        # control_range_scaling=[range[end] - range[1] for range in control_ranges],
    )

    plot_simulation(controlODE, θ; only=:states)
    plot_simulation(controlODE, θ; only=:controls)
    store_simulation("precondition", controlODE, θ; datadir)

    function state_penalty_functional(solution_array, time_intervals; δ=1.0f1)
        @argcheck size(solution_array, 2) == length(time_intervals) + 1

        ratio_X_N = 3.0f-2 / 800.0f0
        C_N_over = map(y -> relaxed_log_barrier(800.0f0 - y; δ), solution_array[2, 1:end])
        C_X_over = map(
            (x, z) -> relaxed_log_barrier(3.0f-2 - (1.1f-2 * x - z); δ=δ * ratio_X_N),
            solution_array[1, 1:end],
            solution_array[3, 1:end],
        )
        C_N_over_last = relaxed_log_barrier(150.0f0 - solution_array[2, end]; δ)

        return sum((C_N_over .+ C_X_over)[2:end] .* time_intervals) + C_N_over_last
    end

    # state constraints on control change
    # C_N(t) - 150 ≤ 0              t = T
    # C_N(t) − 800 ≤ 0              ∀t
    # 0.011 C_X(t) - C_qc(t) ≤ 3f-2 ∀t
    function losses(
        controlODE,
        params;
        δ=1.0f1,
        α=1.0f-5,
        μ=(3.125f-8, 3.125f-6),
        ρ=1.0f-1,
        tsteps=(),
        kwargs...,
    )

        # integrate ODE system
        sol_raw = solve(controlODE, params; kwargs...)
        sol = Array(sol_raw)

        # approximate integral penalty
        # state_penalty = Δt * (sum(C_N_over) + sum(C_X_over)) + C_N_over_last  # for fixed timesteps
        Zygote.ignore() do
            global time_intervals = [
                sol_raw.t[i + 1] - sol_raw.t[i] for i in eachindex(sol_raw.t[1:(end - 1)])
            ]
        end

        state_penalty = α * state_penalty_functional(sol, time_intervals; δ)

        # # penalty on change of controls
        control_penalty = 0.0f0
        # for i in 1:(size(sol, 2) - 1)
        #     prev = controller(sol[:, i], params)
        #     post = controller(sol[:, i + 1], params)
        #     for j in 1:length(prev)
        #         control_penalty += μ[j] * (prev[j] - post[j])^2
        #     end
        # end

        regularization = ρ * mean(abs2, θ)  # sum(abs2, θ)

        objective = -sol[3, end]  # maximize C_qc

        return objective, state_penalty, control_penalty, regularization
    end

    function plots_callback(controlODE, θ)
        plot_simulation(controlODE, θ; only=:states, vars=[2], yrefs=[800, 150])
        plot_simulation(
            controlODE, θ; only=:states, fun=(x, y, z) -> 1.1f-2x - z, yrefs=[3.0f-2]
        )
        return plot_simulation(controlODE, θ; only=:controls)
    end

    # α: penalty coefficient
    # δ: barrier relaxation coefficient
    α0, δ0 = 1.0f-5, 100.0f0
    barrier_iterations = 0:20
    αs = [α0 for _ in barrier_iterations]
    δs = [δ0 * 0.8f0^i for i in barrier_iterations]
    θ = constrained_training(
        controlODE,
        losses;
        αs,
        δs,
        starting_params=θ,
        show_progressbar=true,
        # plots_callback,
        datadir,
    )

    final_values = NamedTuple{(
        :objective, :state_penalty, :control_penalty, :regularization
    )}(
        loss(θ; δ=δs[end], α=αs[end], controlODE.tsteps)
    )

    @info "Final states"
    # plot_simulation(controlODE, θ; only=:states, vars=[1], show=final_values)
    plot_simulation(controlODE, θ; only=:states, vars=[2], yrefs=[800, 150])
    plot_simulation(controODE, θ; only=:states, vars=[3])
    plot_simulation(
        controlODE, θ; only=:states, fun=(x, y, z) -> 1.1f-2x - z, yrefs=[3.0f-2]
    )

    @info "Final controls"
    plot_simulation(controODE, θ; only=:controls, vars=[1])
    plot_simulation(controODE, θ; only=:controls, vars=[2])

    @info "Final loss" loss(θ; δ=δs[end], α=αs[end], controlODE.tsteps)

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
        (variable=1, type=:centered, scale=10.0f0, samples=10, percentage=2.0f-2)
        (variable=2, type=:centered, scale=800.0f0, samples=10, percentage=2.0f-2)
        (variable=3, type=:positive, scale=5.0f-1, samples=10, percentage=2.0f-2)
    ]
    return plot_initial_perturbations(controlODE, θ, perturbation_specs)
end  # script wrapper
