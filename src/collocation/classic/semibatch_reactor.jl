function semibatch_reactor_collocation(
    u0=controlODE.u0,
    tspan=controlODE.tspan;
    num_supports::Integer=length(controlODE.tsteps),
    nodes_per_element::Integer=3,
    constrain_states::Bool=false,
)
    optimizer = optimizer_with_attributes(
        Ipopt.Optimizer, "print_level" => 0, "check_derivatives_for_naninf" => "yes"
    )
    model = InfiniteModel(optimizer)
    method = OrthogonalCollocation(nodes_per_element)

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

    t0, tf = tspan
    @infinite_parameter(
        model, t in [t0, tf], num_supports = num_supports, derivative_method = method
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

    jump_model = optimizer_model(model)
    solution_summary(jump_model; verbose=false)

    return extract_infopt_results(model)
end
