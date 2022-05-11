# Bradford, E., Imsland, L., Zhang, D., & del Rio Chanona, E. A. (2020).
# Stochastic data-driven model predictive control using Gaussian processes.
# Computers & Chemical Engineering, 139, 106844.

# objective: maximize C_qc

# state constraints
# C_N(t) - 150 ≤ 0              t = T
# C_N(t) − 400 ≤ 0              ∀t
# 0.011 C_X(t) - C_qc(t) ≤ 3f-2 ∀t

function bioreactor(; store_results::Bool=false)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    # initial conditions and timepoints
    t0 = 0.0
    tf = 240.0
    Δt = 10.0
    tspan = (t0, tf)
    C_X₀, C_N₀, C_qc₀ = 1.0, 150.0, 0.0
    u0 = [C_X₀, C_N₀, C_qc₀]
    control_ranges = [(80.0, 180.0), (0.0, 20.0)]

    # set arquitecture of neural network controller
    # weights initializer reference https://pytorch.org/docs/stable/nn.init.html
    controller = FastChain(
        (x, p) -> [x[1], x[2] / 100.0, x[3] * 10.0],  # input scaling
        FastDense(3, 16, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        FastDense(16, 16, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        FastDense(16, 2; initW=(x, y) -> glorot_uniform(x, y)),
        # I ∈ [120, 400] & F ∈ [0, 40] in Bradford 2020
        # (x, p) -> [280f0 * sigmoid(x[1]) + 120f0, 40f0 * sigmoid(x[2])],
        scaled_sigmoids(control_ranges),
    )

    system = BioReactor()
    controlODE = ControlODE(controller, system, u0, tspan; Δt)

    function plot_state_constraints(θ)
        plot_simulation(controlODE, θ; only=:states, vars=[2], yrefs=[400, 250])
        plot_simulation(controlODE, θ; only=:states, vars=[3])
    end

    collocation = bioreactor_collocation(
        controlODE.u0,
        controlODE.tspan;
        constrain_states=true,
    )

    reference_controller = interpolant_controller(collocation; plot=:unicode)

    θ = preconditioner(
        controlODE,
        reference_controller;
        θ=Float64.(initial_params(controller)),
        ## Flux options
        # optimizer=ADAM(),
        x_tol=1e-5,
        ## Optim options
        # optimizer=LBFGS(; linesearch=BackTracking()),
        # iterations=100,
        # x_tol=1.0f-2,
        # f_tol=1.0f-2,
    )

    # θ = initial_params(controlODE.controller)

    plot_state_constraints(θ)
    plot_simulation(controlODE, θ; only=:controls, vars=[1])
    plot_simulation(controlODE, θ; only=:controls, vars=[2])
    store_simulation("precondition", controlODE, θ; datadir)

    function state_penalty_functional(solution_array; δ)

        C_N_over_running = map(y -> relaxed_log_barrier(400.0 - y; δ), solution_array[2, 1:end-1])
        C_N_over_last = relaxed_log_barrier(solution_array[2, end] - 250.0; δ)

        Δt = Float32(controlODE.tsteps.step)
        return Δt * sum(C_N_over_running) + C_N_over_last
    end

    # state constraints on control change
    # C_N(t) - 250 ≥ 0              t = T
    # C_N(t) − 400 ≤ 0              ∀t
    function losses(controlODE, params; α, δ, ρ)
        # integrate ODE system
        sol_raw = solve(controlODE, params)
        sol = Array(sol_raw)

        # https://diffeqflux.sciml.ai/dev/examples/divergence/
        # Zygote.@ignore @infiltrate sol_raw.retcode != :Success
        # if sol_raw.retcode != :Success  # avoid this with Zygote...
        if sol_raw.t[end] != tf
            # Zygote.@ignore @infiltrate
            return Inf
        end

        objective = -sol[3, end]  # maximize C_qc

        state_penalty = α * state_penalty_functional(sol; δ)

        # # penalty on change of controls
        control_penalty = 0.0
        # for i in 1:(size(sol, 2) - 1)
        #     prev = controller(sol[:, i], params)
        #     post = controller(sol[:, i + 1], params)
        #     for j in 1:length(prev)
        #         control_penalty += μ[j] * (prev[j] - post[j])^2
        #     end
        # end

        regularization = ρ * sum(abs2, params)
        # regularization = 0.0

        # Zygote.@ignore @infiltrate abs(regularization) > 1.0f2 &&
        #     abs(regularization) >
        #                            1.0f1 * (abs(objective) + abs(state_penalty))
        return objective, state_penalty, control_penalty, regularization
    end

    # α: penalty coefficient
    # δ: barrier relaxation coefficient
    α = 1f-4
    ρ = 1f-2
    δ0 = 2f0
    δ_final = 1f-1 * δ0
    max_barrier_iterations = 100
    # Zygote.@ignore @infiltrate
    # return
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
        # optimizer=LBFGS(; linesearch=BackTracking()),
        # iterations=50,
        # ## Flux options
        # optimizer=ADAM(1f-1),
        # x_tol=1.0f-5,
        # f_tol=1.0f-2,
        # g_tol=1f-1,
        # maxiters=100,
    )

    objective, state_penalty, control_penalty, regularization = losses(
        controlODE, θ; δ, α, ρ
    )

    @info "Final states"
    # plot_simulation(controlODE, θ; only=:states, vars=[1], show=final_values)
    plot_state_constraints(θ)

    @info "Final controls"
    plot_simulation(controlODE, θ; only=:controls, vars=[1])
    plot_simulation(controlODE, θ; only=:controls, vars=[2])

    @info "Final losses" losses(controlODE, θ; δ, α, ρ)

    @info "Collocation comparison"
    collocation = bioreactor_collocation(
        controlODE.u0,
        controlODE.tspan;
        # num_supports=length(controlODE.tsteps),
        # nodes_per_element=2,
        constrain_states=true,
    )
    interpolant_controller(collocation; plot=:unicode)

    # perturbation_specs = [
    #     (variable=1, type=:centered, scale=1.0f0, samples=3, percentage=5.0f-2)
    #     (variable=2, type=:centered, scale=400.0f0, samples=3, percentage=5.0f-2)
    #     (variable=3, type=:positive, scale=5.0f-1, samples=3, percentage=5.0f-2)
    # ]
    # plot_initial_perturbations(controlODE, θ, perturbation_specs)

end  # script wrapper
