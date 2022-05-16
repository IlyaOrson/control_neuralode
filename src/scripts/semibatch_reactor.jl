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
    fixed_controlODE = ControlODE((u, p) -> fogler_ref, system, u0, fogler_timespan; Δt=1.5f-1)
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

    collocation = semibatch_reactor_collocation(
        controlODE.u0,
        controlODE.tspan;
        num_supports=length(controlODE.tsteps),
        nodes_per_element=2,
        constrain_states=false,
    )
    # collocation_constrained = semibatch_reactor_collocation(
    #     controlODE.u0,
    #     controlODE.tspan;
    #     num_supports=length(controlODE.tsteps),
    #     constrain_states=true,
    #     nodes_per_element=2
    # )

    # plt.figure()
    # plt.plot(collocation.times, collocation.states[1, :]; label="s1")
    # plt.plot(collocation.times, collocation.states[2, :]; label="s2")
    # plt.plot(collocation.times, collocation.states[3, :]; label="s3")
    # plt.plot(
    #     collocation.times, collocation_constrained.states[1, :];
    #     label="s1_constrained", color=plt.gca().lines[1].get_color(), ls="dashdot"
    # )
    # plt.plot(
    #     collocation.times, collocation_constrained.states[2, :];
    #     label="s2_constrained", color=plt.gca().lines[2].get_color(), ls="dashdot"
    # )
    # plt.plot(
    #     collocation.times, collocation_constrained.states[3, :];
    #     label="s3_constrained", color=plt.gca().lines[3].get_color(), ls="dashdot"
    # )
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(collocation.times, collocation.states[4, :]; label="s4")
    # plt.plot(collocation.times, collocation.states[5, :]; label="s5")
    # plt.plot(
    #     collocation.times, collocation_constrained.states[4, :];
    #     label="s4_constrained", color=plt.gca().lines[1].get_color(), ls="dashdot"
    # )
    # plt.plot(
    #     collocation.times, collocation_constrained.states[5, :];
    #     label="s5_constrained", color=plt.gca().lines[2].get_color(), ls="dashdot"
    # )
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(collocation.times, collocation.controls[1, :]; label="c1")
    # plt.plot(collocation.times, collocation.controls[2, :]; label="c2")
    # plt.plot(
    #     collocation.times, collocation_constrained.controls[1, :];
    #     label="c1_constrained", color=plt.gca().lines[1].get_color(), ls="dashdot"
    # )
    # plt.plot(
    #     collocation.times, collocation_constrained.controls[2, :];
    #     label="c2_constrained", color=plt.gca().lines[2].get_color(), ls="dashdot"
    # )
    # plt.legend()
    # plt.show()

    reference_controller = interpolant_controller(collocation; plot=nothing)

    θ = preconditioner(
        controlODE,
        reference_controller;
        θ=initial_params(controller),
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
    α = 1f-3
    ρ = 1f-3
    δ0 = 1f1
    max_barrier_iterations = 25
    δ_final = 5f-2 * δ0
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
    )

    @info "Final states"
    plot_simulation(controlODE, θ; only=:states, vars=[1, 2, 3])
    plot_simulation(
        controlODE, θ; only=:states, vars=[4, 5], yrefs=[T_up, V_up]
    )

    @info "Final controls"
    plot_simulation(controlODE, θ; only=:controls)#  only=:states, vars=[1,2,3])

    @info "Final loss" losses(controlODE, θ; δ, α, ρ)

end  # function wrapper
