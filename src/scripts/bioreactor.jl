# Bradford, E., Imsland, L., Zhang, D., & del Rio Chanona, E. A. (2020).
# Stochastic data-driven model predictive control using Gaussian processes.
# Computers & Chemical Engineering, 139, 106844.

# objective: maximize C_qc

# state constraints
# C_N(t) - 150 ≤ 0              t = T
# C_N(t) − 800 ≤ 0              ∀t
# 0.011 C_X(t) - C_qc(t) ≤ 3f-2 ∀t

function bioreactor(; store_results::Bool=false)
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
        FastDense(3, 64, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        FastDense(64, 64, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        # FastDense(64, 64, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        FastDense(64, 2; initW=(x, y) -> glorot_uniform(x, y)),
        # I ∈ [120, 400] & F ∈ [0, 40] in Bradford 2020
        # (x, p) -> [280f0 * sigmoid(x[1]) + 120f0, 40f0 * sigmoid(x[2])],
        scaled_sigmoids(control_ranges),
    )

    system = BioReactor()
    controlODE = ControlODE(controller, system, u0, tspan; Δt)

    function plot_state_constraints(θ)
        plot_simulation(controlODE, θ; only=:states, vars=[2], yrefs=[800, 150])
        plot_simulation(controlODE, θ; only=:states, vars=[3])
        plot_simulation(
            controlODE, θ; only=:states, fun=(x, y, z) -> 1.1f-2x - z, yrefs=[3.0f-2]
        )
    end

    collocation = bioreactor_collocation(
        controlODE.u0,
        controlODE.tspan;
        constrain_states=false,
    )

    reference_controller = interpolant_controller(collocation; plot=false)

    θ = preconditioner(
        controlODE,
        reference_controller;
        ## Optim options
        # optimizer=LBFGS(; linesearch=BackTracking()),
        # iterations=100,
        # x_tol=1.0f-2,
        # f_tol=1.0f-2,
        # ## Flux options
        optimizer=ADAM(),
        # maxiters=50,
    )

    # θ = initial_params(controlODE.controller)

    plot_state_constraints(θ)
    plot_simulation(controlODE, θ; only=:controls, vars=[1])
    plot_simulation(controlODE, θ; only=:controls, vars=[2])
    store_simulation("precondition", controlODE, θ; datadir)

    function state_penalty_functional(solution_array; δ=1.0f1)
        ratio_X_N = 3.0f-2 / 800.0f0
        C_N_over = map(y -> relaxed_log_barrier(800.0f0 - y; δ), solution_array[2, 1:end])
        C_X_over = map(
            (x, z) -> relaxed_log_barrier(3.0f-2 - (1.1f-2 * x - z); δ=δ * ratio_X_N),
            solution_array[1, 1:end],
            solution_array[3, 1:end],
        )
        C_N_over_last = relaxed_log_barrier(150.0f0 - solution_array[2, end]; δ)

        Δt = Float32(controlODE.tsteps.step)
        return Δt * sum((C_N_over .+ C_X_over)[2:end]) + C_N_over_last
    end

    # state constraints on control change
    # C_N(t) - 150 ≤ 0              t = T
    # C_N(t) − 800 ≤ 0              ∀t
    # 0.011 C_X(t) - C_qc(t) ≤ 3f-2 ∀t
    function losses(
        controlODE,
        params;
        α,
        δ,
        # μ=(3.125f-8, 3.125f-6),
        ρ,
        kwargs...,
    )

        # integrate ODE system
        sol_raw = solve(controlODE, params; kwargs...)
        sol = Array(sol_raw)

        # https://diffeqflux.sciml.ai/dev/examples/divergence/
        # if sol_raw.retcode != :Success  # avoid this with Zygote...
        if sol_raw.t[end] != tf
            # Zygote.@ignore @infiltrate
            return Inf
        end

        state_penalty = α * state_penalty_functional(sol; δ)
        # state_penalty = 0.0f0

        # # penalty on change of controls
        control_penalty = 0.0f0
        # for i in 1:(size(sol, 2) - 1)
        #     prev = controller(sol[:, i], params)
        #     post = controller(sol[:, i + 1], params)
        #     for j in 1:length(prev)
        #         control_penalty += μ[j] * (prev[j] - post[j])^2
        #     end
        # end

        regularization = ρ * sum(abs2, params)
        # regularization = 0.0f0

        objective = -sol[3, end]  # maximize C_qc
        Zygote.@ignore @infiltrate abs(regularization) > 1.0f2 &&
            abs(regularization) >
                                   1.0f1 * (abs(objective) + abs(state_penalty))
        return objective, state_penalty, control_penalty, regularization
    end

    # α: penalty coefficient
    # δ: barrier relaxation coefficient
    α = 1f-1
    ρ = 1f-3
    δ0 = 1f2
    max_barrier_iterations = 100
    δ_final = 1f-2 * δ0

    θ, δ = constrained_training(
        controlODE,
        losses,
        δ0;
        θ,
        δ_final,
        max_barrier_iterations,
        α,
        ρ,
        show_progressbar=true,
        datadir,
        # Optim options
        optimizer=LBFGS(; linesearch=BackTracking()),
        # iterations=50,
        # ## Flux options
        # optimizer=ADAM(),
        # x_tol=1.0f-6,
        # f_tol=1.0f-2,
        # maxiters=100,
    )

    objective, state_penalty, control_penalty, regularization = losses(
        controlODE, θ; δ, α, ρ, tsteps=controlODE.tsteps
    )

    @info "Final states"
    # plot_simulation(controlODE, θ; only=:states, vars=[1], show=final_values)
    plot_state_constraints(θ)

    @info "Final controls"
    plot_simulation(controlODE, θ; only=:controls, vars=[1])
    plot_simulation(controlODE, θ; only=:controls, vars=[2])

    @info "Final losses" losses(controlODE, θ; δ, α, ρ, controlODE.tsteps)

    @info "Collocation comparison"
    collocation = bioreactor_collocation(
        controlODE.u0,
        controlODE.tspan;
        # num_supports=length(controlODE.tsteps),
        # nodes_per_element=2,
        constrain_states=true,
    )
    interpolant_controller(collocation; plot=true)

    # @infiltrate

    # initial conditions and timepoints
    # t0 = 0f0
    # tf = 240f0
    # Δt = 10f0
    # C_X₀, C_N₀, C_qc₀ = 1f0, 150f0, 0f0
    # u0 = [C_X₀, C_N₀, C_qc₀]
    # tspan = (t0, tf)
    # tsteps = t0:Δt:tf
    # control_ranges = [(120f0, 400f0), (0f0, 40f0)]

    # perturbation_specs = [
    #     (variable=1, type=:centered, scale=1.0f0, samples=3, percentage=5.0f-2)
    #     (variable=2, type=:centered, scale=800.0f0, samples=3, percentage=5.0f-2)
    #     (variable=3, type=:positive, scale=5.0f-1, samples=3, percentage=5.0f-2)
    # ]
    # plot_initial_perturbations(controlODE, θ, perturbation_specs)

end  # script wrapper
