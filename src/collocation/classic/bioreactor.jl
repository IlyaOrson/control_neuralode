function bioreactor_collocation(
    u0,
    tspan;
    num_supports::Integer=25,
    nodes_per_element::Integer=2,
    constrain_states::Bool=false,
)
    t0, tf = tspan
    optimizer = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 2,
    )
    model = InfiniteModel(optimizer)
    method = OrthogonalCollocation(nodes_per_element)
    @infinite_parameter(
        model,
        t in [t0, tf],
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
            80 <= c[1] <= 180
            0 <= c[2] <= 20
        end
    )

    if constrain_states
        @constraints(
            model,
            begin
                x[2](tf - 1e-2) >= 250
                x[2] <= 400
                # 1.1e-2 * x[1] - x[3] <= 1.0e-2
            end
        )
    end

    # dynamic equations
    @constraints(
        model,
        begin
            ∂(x[1], t) ==
            u_m *
            (c[1] / (c[1] + k_s + c[1]^2.0 / k_i)) *
            x[1] *
            (x[2] / (x[2] + K_N)) - u_d * x[1]
            ∂(x[2], t) ==
            -Y_NX *
            u_m *
            (c[1] / (c[1] + k_s + c[1]^2.0 / k_i)) *
            x[1] *
            (x[2] / (x[2] + K_N)) + c[2]
            ∂(x[3], t) ==
            k_m * (c[1] / (c[1] + k_sq + c[1]^2.0 / k_iq)) * x[1] -
            k_d * (x[3] / (x[2] + K_Np))
        end
    )

    @objective(model, Max, x[3](tf))

    optimize_infopt!(model)

    jump_model = optimizer_model(model)
    solution_summary(jump_model; verbose=false)

    return model
end
