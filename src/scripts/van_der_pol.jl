# Vassiliadis, V. S., Sargent, R. W. H., & Pantelides, C. C. (1994).
# Solution of a Class of Multistage Dynamic Optimization Problems. 2. Problems with Path Constraints.
# Industrial & Engineering Chemistry Research, 33(9), 2123–2133. https://doi.org/10.1021/ie00033a015

function van_der_pol(; store_results::Bool=false)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    system = VanDerPol()
    controlODE = ControlODE(system)

    θ = initial_params(controlODE.controller)

    _, states_raw, _ = run_simulation(controlODE, θ)
    phase_portrait(
        controlODE,
        θ,
        square_bounds(controlODE.u0, 7);
        projection=[1, 2],
        markers=states_markers(states_raw),
        title="Initial policy",
    )

    collocation = van_der_pol_collocation(
        controlODE.u0,
        controlODE.tspan;
        num_supports=length(controlODE.tsteps),
        nodes_per_element=2,
        constrain_states=false,
    )
    reference_controller = interpolant_controller(collocation; plot=nothing)

    θ = preconditioner(
        controlODE,
        reference_controller;
        θ,
        x_tol=1f-7,
        g_tol=1f-2,
    )

    plot_simulation(controlODE, θ; only=:controls)
    store_simulation("precondition", controlODE, θ; datadir)

    _, states_raw, _ = run_simulation(controlODE, θ)
    phase_portrait(
        controlODE,
        θ,
        square_bounds(controlODE.u0, 3);
        projection=[1, 2],
        markers=states_markers(states_raw),
        title="Preconditioned policy",
    )

    ### define objective function to optimize
    function loss(controlODE, params; kwargs...)
        sol = solve(controlODE, params) |> Array
        # return Array(sol)[3, end]  # return last value of third variable ...to be minimized
        objective = 0.0f0
        for i in axes(sol, 2)
            s = sol[:, i]
            c = controlODE.controller(s, params)
            objective += s[1]^2 + s[2]^2 + c[1]^2
        end
        return objective
    end
    loss(params) = loss(controlODE, params)

    @info "Training..."
    grad!(g, params) = g .= Zygote.gradient(loss, params)[1]
    # θ = optimize_optim(θ, loss, grad!)
    @infiltrate
    θ = optimize_ipopt(θ, loss, grad!)

    _, states_raw, _ = run_simulation(controlODE, θ)
    phase_portrait(
        controlODE,
        θ,
        square_bounds(controlODE.u0, 3);
        projection=[1, 2],
        markers=states_markers(states_raw),
        title="Optimized policy",
    )

    store_simulation(
        "unconstrained",
        controlODE,
        θ;
        datadir,
        metadata=Dict(:loss => loss(θ), :constraint => "none"),
    )

    ### now add state constraint x1(t) > -0.4 with
    function losses(controlODE, params; α, δ, ρ)
        # integrate ODE system
        Δt = Float32(controlODE.tsteps.step)
        sol = solve(controlODE, params) |> Array
        objective = 0.0f0
        control_penalty = 0.0f0
        for i in axes(sol, 2)
            s = sol[:, i]
            c = controlODE.controller(s, params)
            objective += s[1]^2 + s[2]^2
            control_penalty += c[1]^2
        end
        objective *= Δt
        control_penalty *= Δt

        # fault = min.(sol[1, 1:end] .+ 0.4f0, 0.0f0)
        state_fault = map(x -> relaxed_log_barrier(x - -0.4f0; δ), sol[1, 1:end-1])
        # penalty = α * sum(fault .^ 2)  # quadratic penalty
        state_penalty = Δt * α * sum(state_fault)
        regularization = ρ * sum(abs2, params)
        return objective, state_penalty, control_penalty, regularization
    end

    @info "Enforcing constraints..."
    # α: penalty coefficient
    # δ: barrier relaxation coefficient
    α = 1f-1
    ρ = 0f0
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
    # penalty_loss(result.minimizer, constrained_prob, tsteps; α=penalty_coefficients[end])
    plot_simulation(controlODE, θ; only=:controls)

    objective, state_penalty, control_penalty, regularization = losses(controlODE, θ; α, δ = δ_final, ρ)
    store_simulation(
        "constrained",
        controlODE,
        θ;
        datadir,
        metadata=Dict(
            # :loss => penalty_loss(controlODE, θ; α=α0, δ),
            # :constraint => "quadratic x2(t) > -0.4",
            :objective => objective,
            :state_penalty => state_penalty,
            :control_penalty => control_penalty,
            :regularization => regularization,
        ),
    )

    _, states_opt, _ = run_simulation(controlODE, θ)
    function indicator(coords...)
        if coords[1] > -0.4
            return true
        end
        return false
    end
    shader = ShadeConf(; indicator)
    phase_portrait(
        controlODE,
        θ,
        square_bounds(controlODE.u0, 3);
        shader,
        projection=[1, 2],
        markers=states_markers(states_opt),
        title="Optimized policy with constraints",
    )

    # u0 = [0f0, 1f0]
    # perturbation_specs = [
    #     (variable=1, type=:positive, scale=1.0f0, samples=3, percentage=1.0f-1)
    #     (variable=2, type=:negative, scale=1.0f0, samples=3, percentage=1.0f-1)
    #     # (variable=3, type=:positive, scale=20.0f0, samples=8, percentage=2.0f-2)
    # ]
    # constraint_spec = ConstRef(; val=-0.4, direction=:horizontal, class=:state, var=1)

    # plot_initial_perturbations_collocation(
    #     controlODE,
    #     θ,
    #     perturbation_specs,
    #     van_der_pol_collocation;
    #     refs=[constraint_spec],
    #     storedir=datadir,
    # )
    return
end
