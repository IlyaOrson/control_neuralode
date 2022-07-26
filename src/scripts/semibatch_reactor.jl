# Elements of Chemical Reaction Engineering
# Fifth Edition
# H. SCOTT FOGLER
# Chapter 13: Unsteady-State Nonisothermal Reactor Design
# Section 13.5: Nonisothermal Multiple Reactions
# Example 13–5 Multiple Reactions in a Semibatch Reactor
# p. 658

function semibatch_reactor(; store_results::Bool=false)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    # state constraints
    # T ∈ (0, 420]
    # Vol ∈ (0, 200]
    T_up = 380.0f0
    V_up = 100.0f0

    system = SemibatchReactor()
    controlODE = ControlODE(system)

    # simulate the system with constant controls as in Fogler's
    # to reproduce his results and verify correctness
    fogler_ref = [240.0f0, 298.0f0]  # reference values in Fogler
    fogler_timespan = (0.0f0, 1.5f0)
    fixed_controlODE = ControlODE((u, p) -> fogler_ref, system, controlODE.u0, fogler_timespan; Δt=1.5f-1)
    @info "Fogler's case: final time state" solve(fixed_controlODE, nothing).u[end]
    plot_simulation(
        fixed_controlODE,
        nothing;
        only=:states,
        vars=[1, 2, 3],
    )
    plot_simulation(
        fixed_controlODE,
        nothing;
        only=:states,
        vars=[4, 5],
    )

    collocation_model = semibatch_reactor_collocation(
        controlODE.u0,
        controlODE.tspan;
        num_supports=length(controlODE.tsteps),
        nodes_per_element=2,
        constrain_states=false,
    )
    collocation_results = extract_infopt_results(collocation_model)

    reference_controller = interpolant_controller(collocation_results; plot=nothing)

    θ = initial_params(controlODE.controller)
    θ = preconditioner(
        controlODE,
        reference_controller;
        θ,
        x_tol=nothing,
        f_tol=1.0f-3,
        maxiters=2_000,
    )
    plot_simulation(controlODE, θ; only=:states)
    plot_simulation(controlODE, θ; only=:controls)
    store_simulation("precondition", controlODE, θ; datadir)

    # objective function splitted componenets to optimize
    function losses(controlODE, params; α, δ, ρ)

        # integrate ODE system
        sol_raw = solve(controlODE, params)
        sol_array = Array(sol_raw)

        # https://diffeqflux.sciml.ai/dev/examples/divergence/
        # if sol_raw.retcode != :Success  # avoid this with Zygote...
        Zygote.@ignore if sol_raw.t[end] != controlODE.tspan[end]
            return Inf
        end

        # running cost
        out_temp = map(x -> relaxed_log_barrier(T_up - x; δ), sol_array[4, 1:end])
        out_vols = map(x -> relaxed_log_barrier(V_up - x; δ), sol_array[5, 1:end])

        # terminal cost
        # L = - (100 x₁ - x₂) + penalty  # Bradford
        objective = -sol_array[2, end]

        # integral penalty
        Δt = Float32(controlODE.tsteps.step)
        state_penalty = α * Δt * (sum(out_temp) + sum(out_vols))
        control_penalty = 0.0f0
        regularization = ρ * sum(abs2, params)
        return (; objective, state_penalty, control_penalty, regularization)
    end

    # α: penalty coefficient
    # ρ: regularization coefficient
    # δ: barrier relaxation coefficient
    ρ = 1f-3
    θ, barrier_progression = constrained_training(
        losses,
        controlODE,
        θ;
        ρ,
        show_progressbar=true,
        datadir,
    )
    @info "Alpha progression" barrier_progression.α
    lineplot(log.(barrier_progression.α))

    @info "Delta progression" barrier_progression.δ
    lineplot(log.(barrier_progression.δ))

    δ_final = barrier_progression.δ[end]
    α_final = barrier_progression.α[end]
    objective, state_penalty, control_penalty, regularization = losses(
        controlODE, θ; δ = δ_final, α = α_final, ρ
    )

    @info "Final states"
    plot_simulation(controlODE, θ; only=:states, vars=[1, 2, 3])
    plot_simulation(
        controlODE, θ; only=:states, vars=[4, 5], yrefs=[T_up, V_up]
    )

    @info "Final controls"
    plot_simulation(controlODE, θ; only=:controls)#  only=:states, vars=[1,2,3])

    @info "Final losses" objective state_penalty control_penalty regularization

    @info "Collocation comparison"
    collocation_model = semibatch_reactor_collocation(
        controlODE.u0,
        controlODE.tspan;
        constrain_states=true,
    )
    collocation_results = extract_infopt_results(collocation_model)
    interpolant_controller(collocation_results)

    @info "Collocation states"
    for states in eachrow(collocation_results.states)
        lineplot(states) |> display
    end

    @info "Collocation controls"
    for controls in eachrow(collocation_results.controls)
        lineplot(controls) |> display
    end

end  # function wrapper
