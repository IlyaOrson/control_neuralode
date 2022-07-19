# Bradford, E., Imsland, L., Zhang, D., & del Rio Chanona, E. A. (2020).
# Stochastic data-driven model predictive control using Gaussian processes.
# Computers & Chemical Engineering, 139, 106844.

# objective: maximize C_qc

# state constraints
# C_N(t) - 250 ≥ 0      t = T
# C_N(t) − 400 ≤ 0      ∀t

function bioreactor(; store_results::Bool=false)

    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    system = BioReactor()
    controlODE = ControlODE(system)

    function plot_state_constraints(θ)
        plot_simulation(controlODE, θ; only=:states, vars=[2], yrefs=[400, 250])
        plot_simulation(controlODE, θ; only=:states, vars=[3])
    end

    collocation_model = bioreactor_collocation(
        controlODE.u0,
        controlODE.tspan;
        constrain_states=true,
    )
    collocation_results = extract_infopt_results(model)
    reference_controller = interpolant_controller(collocation; plot=:unicode)

    θ = initial_params(controlODE.controller)
    θ = preconditioner(
        controlODE,
        reference_controller;
        θ,
        x_tol=1e-4,
    )

    plot_state_constraints(θ)
    plot_simulation(controlODE, θ; only=:controls, vars=[1])
    plot_simulation(controlODE, θ; only=:controls, vars=[2])
    store_simulation("precondition", controlODE, θ; datadir)

    function state_penalty_functional(solution_array; δ)
        # state constraints
        # C_N(t) - 250 ≥ 0              t = T
        # C_N(t) − 400 ≤ 0              ∀t
        C_N_over_running = map(y -> relaxed_log_barrier(400.0 - y; δ), solution_array[2, 1:end-1])
        C_N_over_last = relaxed_log_barrier(solution_array[2, end] - 250.0; δ)

        # Δt = Float32(controlODE.tsteps.step)
        # return Δt * sum(C_N_over_running) + C_N_over_last
        return mean(C_N_over_running) + C_N_over_last
    end

    function losses(controlODE, params; α, δ, ρ)
        # integrate ODE system
        sol_raw = solve(controlODE, params)
        sol = Array(sol_raw)

        # https://diffeqflux.sciml.ai/dev/examples/divergence/
        # Zygote.@ignore @infiltrate sol_raw.retcode != :Success
        if sol_raw.t[end] != controlODE.tspan[end]
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

        # Zygote.@ignore @infiltrate abs(regularization) > 1.0f2 &&
        #     abs(regularization) >
        #                            1.0f1 * (abs(objective) + abs(state_penalty))
        return objective, state_penalty, control_penalty, regularization
    end

    # α: penalty coefficient
    # δ: barrier relaxation coefficient
    α = 1f-3
    ρ = 1f-2
    θ, δ_progression = constrained_training(
        losses,
        controlODE,
        θ;
        α,
        ρ,
        show_progressbar=true,
        datadir,
    )

    @info "Delta progression" δ_progression

    δ_final = δ_progression[end]
    objective, state_penalty, control_penalty, regularization = losses(
        controlODE, θ; δ = δ_final, α, ρ
    )

    @info "Final states"
    plot_state_constraints(θ)

    @info "Final controls"
    plot_simulation(controlODE, θ; only=:controls, vars=[1])
    plot_simulation(controlODE, θ; only=:controls, vars=[2])

    @info "Final losses" losses(controlODE, θ; δ = δ_final, α, ρ)

    @info "Collocation comparison"
    collocation_model = bioreactor_collocation(
        controlODE.u0,
        controlODE.tspan;
        # num_supports=length(controlODE.tsteps),
        # nodes_per_element=2,
        constrain_states=true,
    )
    collocation_results = extract_infopt_results(collocation_model)
    interpolant_controller(collocation_results; plot=:unicode)

    # perturbation_specs = [
    #     (variable=1, type=:centered, scale=1.0f0, samples=3, percentage=5.0f-2)
    #     (variable=2, type=:centered, scale=400.0f0, samples=3, percentage=5.0f-2)
    #     (variable=3, type=:positive, scale=5.0f-1, samples=3, percentage=5.0f-2)
    # ]
    # plot_initial_perturbations(controlODE, θ, perturbation_specs)

end  # script wrapper
