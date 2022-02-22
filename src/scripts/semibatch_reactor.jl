# Elements of Chemical Reaction Engineering
# Fifth Edition
# H. SCOTT FOGLER
# Chapter 13: Unsteady-State Nonisothermal Reactor Design
# Section 13.5: Nonisothermal Multiple Reactions
# Example 13–5 Multiple Reactions in a Semibatch Reactor
# p. 658

function semibatch_reactor(; store_results=false::Bool)
    datadir = nothing
    if store_results
        datadir = generate_data_subdir(@__FILE__)
    end

    # initial conditions and timepoints
    t0 = 0.0f0
    tf = 0.4f0  # Bradfoard uses 0.4
    Δt = 0.01f0
    tspan = (t0, tf)

    # state: CA, CB, CC, T, Vol
    u0 = [1.0f0, 0.0f0, 0.0f0, 290.0f0, 100.0f0]

    # control constraints
    # F = volumetric flow rate
    # V = exchanger temperature
    # F = 240 & V = 298 in Fogler's book
    # F ∈ (0, 250) & V ∈ (200, 500) in Bradford 2017
    control_ranges = [(100.0f0, 700.0f0), (0.0f0, 400.0f0)]

    # state constraints
    # T ∈ (0, 420]
    # Vol ∈ (0, 200]
    T_up = 380.0f0
    V_up = 100.0f0

    system! = SemibatchReactor()

    # simulate the system with constant controls as in Fogler's
    # to reproduce his results and verify correctness
    fogler_ref = [240.0f0, 298.0f0]  # reference values in Fogler
    fogler_timespan = (0.0f0, 1.5f0)
    fixed_controlODE = ControlODE((u, p) -> fogler_ref, system!, u0, fogler_timespan; Δt)
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

    # set arquitecture of neural network controller
    controller = FastChain(
        (x, p) -> [1f2x[1], 1f2x[2], 1f2x[3], x[4], x[5]],
        FastDense(5, 12, tanh_fast),
        FastDense(12, 12, tanh_fast),
        FastDense(12, 2),
        # (x, p) -> [240f0, 298f0],
        scaled_sigmoids(control_ranges),
    )
    controlODE = ControlODE(controller, system!, u0, tspan; Δt)

    function infopt_collocation(;
        u0=controlODE.u0,
        tspan=controlODE.tspan,
        num_supports::Integer=length(controlODE.tsteps),
        nodes_per_element::Integer=3,
        constrain_states::Bool=false,
    )
        optimizer = optimizer_with_attributes(
            Ipopt.Optimizer, "print_level" => 0, "check_derivatives_for_naninf" => "yes"
        )
        model = InfiniteModel(optimizer)
        method = OrthogonalCollocation(nodes_per_element)
        @infinite_parameter(
            model, t in [tspan[1], tspan[2]], num_supports = num_supports, derivative_method = method
        )

        @variables(
            model,
            begin
                # state variables
                x[1:5], Infinite(t)
                # control variables
                c[1:2], Infinite(t)
            end
        )

        # tricks to make IPOPT work...
        almost_zero = 1.0f-3
        close_to_infinity = 1.0f3

        set_start_value_function(x[1], t -> 0.5f0)
        set_start_value_function(x[2], t -> 0.5f0)
        set_start_value_function(x[3], t -> 0.5f0)
        set_start_value_function(x[4], t -> 2.0f2)
        set_start_value_function(x[5], t -> 1.0f2)

        set_start_value_function(c[1], t -> 2.0f2)
        set_start_value_function(c[2], t -> 1.0f2)

        @constraints(
            model,
            begin
                almost_zero <= x[1] <= close_to_infinity
                almost_zero <= x[2] <= close_to_infinity
                almost_zero <= x[3] <= close_to_infinity
                almost_zero <= x[4] <= close_to_infinity
                almost_zero <= x[5] <= close_to_infinity
            end
        )

        # fixed_parameters
        (; CpA, CpB, CpC, CpH2SO4, N0H2S04, T0, CA0, HRA, HRB, E1A, E2B, A1, A2, UA, Tr1, Tr2) = SemibatchReactor()

        # initial_conditions
        @constraints(
            model,
            begin
                x[1](0) == u0[1]
                x[2](0) == u0[2] + almost_zero
                x[3](0) == u0[3] + almost_zero
            end
        )

        # control_constraints
        @constraints(
            model,
            begin
                control_ranges[1][1] <= c[1] <= control_ranges[1][2]
                control_ranges[2][1] <= c[2] <= control_ranges[2][2]
            end
        )

        if constrain_states
            # state_constraints
            @constraints(
                model,
                begin
                    x[4] <= T_up
                    x[5] <= V_up
                end
            )
        end

        # dynamic_constraints
        @constraints(
            model,
            begin
                ∂(x[1], t) ==
                -(A1 * exp(E1A * ((1.0f0 / Tr1) - (1.0f0 / x[4]))) * x[1]) +
                (CA0 - x[1]) * c[1] / x[5]
                ∂(x[2], t) ==
                0.5f0 * (A1 * exp(E1A * ((1.0f0 / Tr1) - (1.0f0 / x[4]))) * x[1]) -
                (A2 * exp(E2B * ((1.0f0 / Tr2) - (1.0f0 / x[4]))) * x[2]) -
                x[2] * c[1] / x[5]
                ∂(x[3], t) ==
                3.0f0 * (A2 * exp(E2B * ((1.0f0 / Tr2) - (1.0f0 / x[4]))) * x[2]) -
                x[3] * c[1] / x[5]
                ∂(x[4], t) ==
                (
                    UA * (c[2] - x[4]) - CA0 * c[1] * CpA * (x[4] - T0) +
                    (
                        HRA * (-(A1 * exp(E1A * ((1.0f0 / Tr1) - (1.0f0 / x[4]))) * x[1])) +
                        HRB * (-(A2 * exp(E2B * ((1.0f0 / Tr2) - (1.0f0 / x[4]))) * x[2]))
                    ) * x[5]
                ) / ((x[1] * CpA + CpB * x[2] + CpC * x[3]) * x[5] + N0H2S04 * CpH2SO4)
                ∂(x[5], t) == c[1]
            end
        )

        @objective(model, Max, x[2](tf))

        optimize_infopt!(model)

        times = supports(t)
        states = hcat(value.(x)...) |> permutedims
        controls = hcat(value.(c)...) |> permutedims

        return (; model, times, states, controls)
    end

    collocation = infopt_collocation(constrain_states=false, nodes_per_element=2)
    collocation_constrained = infopt_collocation(constrain_states=true, nodes_per_element=2)


    plt.figure()
    plt.plot(collocation.times, collocation.states[1, :]; label="s1")
    plt.plot(collocation.times, collocation.states[2, :]; label="s2")
    plt.plot(collocation.times, collocation.states[3, :]; label="s3")
    plt.plot(
        collocation.times, collocation_constrained.states[1, :];
        label="s1_constrained", color=plt.gca().lines[1].get_color(), ls="dashdot"
    )
    plt.plot(
        collocation.times, collocation_constrained.states[2, :];
        label="s2_constrained", color=plt.gca().lines[2].get_color(), ls="dashdot"
    )
    plt.plot(
        collocation.times, collocation_constrained.states[3, :];
        label="s3_constrained", color=plt.gca().lines[3].get_color(), ls="dashdot"
    )
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(collocation.times, collocation.states[4, :]; label="s4")
    plt.plot(collocation.times, collocation.states[5, :]; label="s5")
    plt.plot(
        collocation.times, collocation_constrained.states[4, :];
        label="s4_constrained", color=plt.gca().lines[1].get_color(), ls="dashdot"
    )
    plt.plot(
        collocation.times, collocation_constrained.states[5, :];
        label="s5_constrained", color=plt.gca().lines[2].get_color(), ls="dashdot"
    )
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(collocation.times, collocation.controls[1, :]; label="c1")
    plt.plot(collocation.times, collocation.controls[2, :]; label="c2")
    plt.plot(
        collocation.times, collocation_constrained.controls[1, :];
        label="c1_constrained", color=plt.gca().lines[1].get_color(), ls="dashdot"
    )
    plt.plot(
        collocation.times, collocation_constrained.controls[2, :];
        label="c2_constrained", color=plt.gca().lines[2].get_color(), ls="dashdot"
    )
    plt.legend()
    plt.show()

    reference_controller = interpolant_controller(collocation; plot=true)
    θ = preconditioner(
        controlODE,
        reference_controller;
        # control_range_scaling=[range[end] - range[1] for range in control_ranges],
    )

    plot_simulation(controlODE, θ; only=:states)
    plot_simulation(controlODE, θ; only=:controls)
    store_simulation("precondition", controlODE, θ; datadir)

    # objective function splitted componenets to optimize
    function losses(controlODE, params; α=1.0f-3, δ=1.0f1, ρ=1.0f-2, kwargs...)

        # integrate ODE system
        sol_array = Array(solve(controlODE, params; kwargs...))

        # running cost
        out_temp = map(x -> relaxed_log_barrier(T_up - x; δ), sol_array[4, 1:end])
        out_vols = map(x -> relaxed_log_barrier(V_up - x; δ), sol_array[5, 1:end])

        # terminal cost
        # L = - (100 x₁ - x₂) + penalty  # Bradford
        objective = -sol_array[2, end]

        # integral penalty
        state_penalty = α * Δt * (sum(out_temp) + sum(out_vols))
        control_penalty = 0.0f0
        regularization = ρ * mean(abs2, θ)
        return objective, state_penalty, control_penalty, regularization
    end

    # α: penalty coefficient
    # δ: barrier relaxation coefficient
    α0, δ0 = 1.0f-5, 1.0f1
    barrier_iterations = 0:8
    αs = [α0 for _ in barrier_iterations]
    δs = [δ0 * 0.8f0^i for i in barrier_iterations]
    θ = constrained_training(
        controlODE,
        losses;
        starting_params=θ,
        αs,
        δs,
        show_progressbar=true,
        # plots_callback,
        datadir,
    )

    @info "Final states"
    plot_simulation(controlODE, θ; only=:states, vars=[1, 2, 3])
    plot_simulation(
        controlODE, θ; only=:states, vars=[4, 5], yrefs=[T_up, V_up]
    )

    @info "Final controls"
    plot_simulation(controlODE, θ; only=:controls)#  only=:states, vars=[1,2,3])

    @info "Final loss" losses(controlODE, θ; δ=δs[end])

    final_objective, final_state_penalty, final_control_penalty, final_regularization = losses(
        controlODE, θ; δ=δs[end]
    )

    return store_simulation(
        "constrained",
        controlODE,
        θ;
        datadir,
        metadata=Dict(
            :objective => final_objective,
            :state_penalty => final_state_penalty,
            :cotrol_penalty => final_control_penalty,
            :regularization => final_regularization,
            :count_params => length(θ),
            :layers => controller_shape(controller),
            :deltas => δs,
            :t0 => t0,
            :tf => tf,
            :Δt => Δt,
        ),
    )
end  # function wrapper
