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
    tf = 1.0f0  # Bradfoard uses 0.4
    Δt = 0.01f0
    tspan = (t0, tf)
    tsteps = t0:Δt:tf

    # state: CA, CB, CC, T, Vol
    u0 = [1.0f0, 0.0f0, 0.0f0, 290.0f0, 100.0f0]

    # control constraints
    # F = volumetric flow rate
    # V = exchanger temperature
    # F = 240 & V = 298 in Fogler's book
    # F ∈ (0, 250) & V ∈ (200, 500) in Bradford 2017
    control_ranges = [(100.0f0, 500.0f0), (0.0f0, 500.0f0)]

    # state constraints
    # T ∈ (0, 420]
    # Vol ∈ (0, 200]
    T_up = 380.0f0
    V_up = 300.0f0

    system_params = (
        CpA=30.0f0,
        CpB=60.0f0,
        CpC=20.0f0,
        CpH2SO4=35.0f0,
        N0H2S04=100.0f0,
        T0=305.0f0,
        CA0=4.0f0,
        HRA=-6500.0f0,
        HRB=8000.0f0,
        E1A=9500.0f0 / 1.987f0,
        E2B=7000.0f0 / 1.987f0,
        A1=1.25f0,
        A2=0.08f0,
        UA=35000.0f0,  # 45000,  Bradford value
        Tr1=320.0f0,  # 420, Bradford value
        Tr2=290.0f0,  # 400, Bradford value
    )

    function system!(du, u, p, t, controller, input=:state)
        @argcheck input in (:state, :time)

        CpA, CpB, CpC, CpH2SO4, N0H2S04, T0, CA0, HRA, HRB, E1A, E2B, A1, A2, UA, Tr1, Tr2 = values(
            system_params
        )

        # neural network outputs controls taken by the system
        CA, CB, CC, T, Vol = u  # states
        # controls
        if input == :state
            c_F, c_T = controller(u, p)
        elseif input == :time
            c_F, c_T = controller(t, p)
        end
        k1A = A1 * exp(E1A * ((1.0f0 / Tr1) - (1.0f0 / T)))
        k2B = A2 * exp(E2B * ((1.0f0 / Tr2) - (1.0f0 / T)))

        k1CA = k1A * CA
        k2CB = k2B * CB
        F_Vol = c_F / Vol

        ra = -k1CA
        rb = 0.5f0 * k1CA - k2CB
        rc = 3.0f0 * k2CB

        num =
            UA * (c_T - T) - CA0 * c_F * CpA * (T - T0) +
            (HRA * (-k1CA) + HRB * (-k2CB)) * Vol
        den = (CA * CpA + CpB * CB + CpC * CC) * Vol + N0H2S04 * CpH2SO4

        # dynamics of the controlled system
        dCA = ra + (CA0 - CA) * F_Vol
        dCB = rb - CB * F_Vol
        dCC = rc - CC * F_Vol
        dT = num / den
        dVol = c_F

        # update in-place
        @inbounds begin
            du[1] = dCA
            du[2] = dCB
            du[3] = dCC
            du[4] = dT
            du[5] = dVol
        end
    end

    # simulate the system with constant controls as in Fogler's
    # to reproduce his results and verify correctness
    fogler_ref = [240.0f0, 298.0f0]  # reference values in Fogler
    fixed_dudt!(du, u, p, t) = system!(du, u, p, t, (u, p) -> fogler_ref)
    fixed_prob = ODEProblem(fixed_dudt!, u0, (0.0f0, 1.5f0))
    fixed_sol = solve(fixed_prob, Tsit5()) |> Array
    @info "Fogler's case: final time state" fixed_sol[:, end]
    plot_simulation(
        (u, p) -> fogler_ref,
        fixed_prob,
        nothing,
        0.0f0:1.0f-1:1.5f0;
        only=:states,
        vars=[1, 2, 3],
    )
    plot_simulation(
        (u, p) -> fogler_ref,
        fixed_prob,
        nothing,
        0.0f0:1.0f-1:1.5f0;
        only=:states,
        vars=[4, 5],
    )

    function collocation(
        u0;
        num_supports::Integer=length(tsteps),
        nodes_per_element::Integer=3,
        constrain_states::Bool=false,
    )
        optimizer = optimizer_with_attributes(
            Ipopt.Optimizer, "print_level" => 0, "check_derivatives_for_naninf" => "yes"
        )
        model = InfiniteModel(optimizer)
        method = OrthogonalCollocation(nodes_per_element)
        @infinite_parameter(
            model,
            t in [Float64(t0), Float64(tf)],
            num_supports = num_supports,
            derivative_method = method
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
        set_start_value_function(c[2], t -> 2.0f2)

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
        CpA, CpB, CpC, CpH2SO4, N0H2S04, T0, CA0, HRA, HRB, E1A, E2B, A1, A2, UA, Tr1, Tr2 = values(
            system_params
        )

        initial_conditions = @constraints(
            model,
            begin
                x[1](0) == u0[1]
                x[2](0) == u0[2] + almost_zero
                x[3](0) == u0[3] + almost_zero
            end
        )

        control_constraints = @constraints(
            model,
            begin
                control_ranges[1][1] <= c[1] <= control_ranges[1][2]
                control_ranges[2][1] <= c[2] <= control_ranges[2][2]
            end
        )

        if constrain_states
            state_constraints = @constraints(
                model,
                begin
                    x[4] <= T_up
                    x[5] <= V_up
                end
            )
        end

        dynamic_constraints = @constraints(
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

        optimize!(model)
        jump_model = optimizer_model(model)
        # @info raw_status(jump_model)
        # @info termination_status(jump_model)
        @info solution_summary(jump_model; verbose=false)
        states = hcat(value.(x)...) |> permutedims
        controls = hcat(value.(c)...) |> permutedims

        return model, supports(t), states, controls
    end

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(5, 8, tanh_fast),
        FastDense(8, 8, tanh_fast),
        FastDense(8, 2),
        # (x, p) -> [240f0, 298f0],
        scaled_sigmoids(control_ranges),
    )

    # destructure model weights into a vector of parameters
    θ = initial_params(controller)

    dudt!(du, u, p, t) = system!(du, u, p, t, controller)

    control_profile, infopt_model, times_collocation, states_collocation, controls_collocation = collocation_preconditioner(
        u0,
        collocation;
        plot=false,
        num_supports=length(tsteps),
        nodes_per_element=2,
        constrain_states=false,
    )
    _, _, _, states_collocation_constrained, controls_collocation_constrained = collocation_preconditioner(
        u0,
        collocation;
        plot=false,
        num_supports=length(tsteps),
        nodes_per_element=2,
        constrain_states=true,
    )

    plt.figure()
    plt.plot(times_collocation, states_collocation[1, :]; label="s1")
    plt.plot(times_collocation, states_collocation[2, :]; label="s2")
    plt.plot(times_collocation, states_collocation[3, :]; label="s3")
    plt.plot(
        times_collocation, states_collocation_constrained[1, :]; label="s1_constrained"
    )
    plt.plot(
        times_collocation, states_collocation_constrained[2, :]; label="s2_constrained"
    )
    plt.plot(
        times_collocation, states_collocation_constrained[3, :]; label="s3_constrained"
    )
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(times_collocation, states_collocation[4, :]; label="s4")
    plt.plot(times_collocation, states_collocation[5, :]; label="s5")
    plt.plot(
        times_collocation, states_collocation_constrained[4, :]; label="s4_constrained"
    )
    plt.plot(
        times_collocation, states_collocation_constrained[5, :]; label="s5_constrained"
    )
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(times_collocation, controls_collocation[1, :]; label="c1")
    plt.plot(times_collocation, controls_collocation[2, :]; label="c2")
    plt.plot(
        times_collocation, controls_collocation_constrained[1, :]; label="c1_constrained"
    )
    plt.plot(
        times_collocation, controls_collocation_constrained[2, :]; label="c2_constrained"
    )
    plt.legend()
    return plt.show()

    # θ = preconditioner(
    #     controller,
    #     control_profile,
    #     system!,
    #     t0,
    #     u0,
    #     tsteps[2:2:end];
    #     progressbar=true,
    #     plot_progress=false,
    #     # control_range_scaling=[range[end] - range[1] for range in control_ranges],
    # )

    # prob = ODEProblem(dudt!, u0, tspan, θ)

    # plot_simulation(controller, prob, θ, tsteps; only=:states)
    # plot_simulation(controller, prob, θ, tsteps; only=:controls)
    # store_simulation("precondition", controller, prob, θ, tsteps; datadir)

    # # define objective function to optimize
    # function loss(params, prob; α=1.0f-3, δ=1.0f1, ρ=1f-2, tsteps=())

    #     # integrate ODE system
    #     # sensealg = InterpolatingAdjoint(; autojacvec=ZygoteVJP(), checkpointing=true)
    #     sol_array = Array(solve(prob, Tsit5(); p=params, saveat=tsteps))

    #     # running cost
    #     out_temp = map(x -> relaxed_log_barrier(T_up - x; δ), sol_array[4, 1:end])
    #     out_vols = map(x -> relaxed_log_barrier(V_up - x; δ), sol_array[5, 1:end])

    #     # terminal cost
    #     # L = - (100 x₁ - x₂) + penalty  # Bradford
    #     objective = sol_array[2, end]

    #     # integral penalty
    #     state_penalty = α * Δt * (sum(out_temp) + sum(out_vols))
    #     control_penalty = 0f0
    #     regularization = ρ * mean(abs2, θ)
    #     return objective, state_penalty, control_penalty, regularization
    # end

    # # α: penalty coefficient
    # # δ: barrier relaxation coefficient
    # α0, δ0 = 1f-5, 1f1
    # barrier_iterations = 0:10
    # αs = [α0 for _ in barrier_iterations]
    # δs = [δ0 * 0.8f0^i for i in barrier_iterations]
    # θ = constrained_training(
    #     controller,
    #     prob,
    #     θ,
    #     loss;
    #     αs,
    #     δs,
    #     tsteps,
    #     show_progressbar=true,
    #     # plots_callback,
    #     datadir,
    # )

    # @info "Final states"
    # plot_simulation(controller, prob, θ, tsteps; only=:states, vars=[1, 2, 3])
    # plot_simulation(
    #     controller, prob, θ, tsteps; only=:states, vars=[4, 5], yrefs=[T_up, V_up]
    # )

    # @info "Final controls"
    # plot_simulation(controller, prob, θ, tsteps; only=:controls, show=loss)#  only=:states, vars=[1,2,3])

    # @show final_objective, final_penalty = loss(θ, prob; δ=δs[end])

    # return store_simulation(
    #     "constrained",
    #     controller,
    #     prob,
    #     θ,
    #     tsteps;
    #     datadir,
    #     metadata=Dict(
    #         :loss => final_objective + final_penalty,
    #         :objective => final_objective,
    #         :penalty => final_penalty,
    #         :num_params => length(initial_params(controller)),
    #         :layers => controller_shape(controller),
    #         :deltas => δs,
    #         :t0 => t0,
    #         :tf => tf,
    #         :Δt => Δt,
    #     ),
    # )
end  # function wrapper
