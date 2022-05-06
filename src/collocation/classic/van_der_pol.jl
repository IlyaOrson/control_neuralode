function van_der_pol_collocation(
    u0,
    tspan;
    num_supports::Integer=10,
    nodes_per_element::Integer=2,
    constrain_states::Bool=false,
)
    optimizer = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 2,
    )
    model = InfiniteModel(optimizer)
    method = OrthogonalCollocation(nodes_per_element)
    @infinite_parameter(
        model, t in [tspan[1], tspan[2]], num_supports = num_supports, derivative_method = method
    )

    @variables(
        model,
        begin  # "start" sets the initial guess values
            # state variables
            x[1:2], Infinite(t)
            # control variables
            c[1], Infinite(t)
        end
    )

    # initial conditions
    @constraint(model, [i = 1:2], x[i](0) == u0[i])

    # control range
    @constraint(model, -0.3 <= c[1] <= 1.0)

    if constrain_states
        @constraint(model, -0.4 <= x[1])
    end

    # dynamic equations
    @constraints(
        model,
        begin
            ∂(x[1], t) == (1 - x[2]^2) * x[1] - x[2] + c[1]
            ∂(x[2], t) == x[1]
        end
    )

    @objective(model, Min, integral(x[1]^2 + x[2]^2 + c[1]^2, t))

    optimize_infopt!(model)

    jump_model = optimizer_model(model)
    solution_summary(jump_model; verbose=false)

    return extract_infopt_results(model)
end
