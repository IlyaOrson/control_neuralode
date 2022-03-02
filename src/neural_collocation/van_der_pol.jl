function van_der_pol_neural_collocation(
    u0, tspan, layer_sizes, activations; nodes_per_element::Integer=2, time_supports::Integer=10
)
    @argcheck length(layer_sizes) == length(activations)

    nstates = length(u0)
    nparams = count_params(nstates, layer_sizes)
    xavier_weights = start_values_sampler(nstates, layer_sizes)

    function vector_fun(z)
        # @show z typeof(z) z[1:nstates] z[nstates+1:end]
        x = collect(z[1:nstates])
        p = collect(z[(nstates + 1):end])
        return chain(x, p, layer_sizes, activations)
    end

    # NOTE: JUMP does not support vector valued functions
    # https://jump.dev/JuMP.jl/stable/manual/nlp/#User-defined-functions-with-vector-inputs
    function scalar_fun(z...)
        return vector_fun(collect(z))
    end

    optimizer = optimizer_with_attributes(Ipopt.Optimizer)
    model = InfiniteModel(optimizer)
    method = OrthogonalCollocation(nodes_per_element)

    @infinite_parameter(
        model,
        t in [tspan[1], tspan[2]],
        num_supports = time_supports,
        derivative_method = method
    )

    # TODO: remove hard constraints?
    @variable(model, -100 <= p[i=1:nparams] <= 100, start = xavier_weights[i])

    # state variables
    @variables(
        model,
        begin
            x[1:2], Infinite(t)
        end
    )

    @constraint(model, [i = 1:2], x[i](0) == u0[i])

    scalar_fun(vcat(x, p)...)
    # @register scalar_fun(vcat(x, p)...)

    @constraints(
        model,
        begin
            ∂(x[1], t) == (1 - x[2]^2) * x[1] - x[2] + scalar_fun(vcat(x, p)...)[1]
            ∂(x[2], t) == x[1]
        end
    )

    @objective(
        model, Min, integral(x[1]^2 + x[2]^2 + scalar_fun(vcat(x, p)...)[1]^2, t)
    )

    optimize_infopt!(model)

    jump_model = optimizer_model(model)

    solution_summary(jump_model; verbose=false)

    return extract_infopt_results(model)
end
