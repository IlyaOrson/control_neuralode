using InfiniteOpt, Ipopt, ArgCheck
function van_der_pol_direct(; store_results=false::Bool)

    function dense(x, p, in_size, out_size, fun=identity)
        @argcheck length(p) == (in_size + 1) * out_size
        matrix = reshape(p[1:(out_size*in_size)], out_size, in_size)
        biases = p[(out_size*in_size+1):end]  # end = out_size * (in_size + 1)
        # @show size(matrix) size(biases)
        fun.(matrix * x .+ biases)
    end

    function num_params(state_size, layers_sizes)
        in_size = state_size
        sum = 0
        for layer_size in layers_sizes
            # @show in_size layer_size
            sum += (in_size + 1) * layer_size
            in_size = layer_size
        end
        return sum
    end

    function chain(x, p, sizes, funs=[tanh for i in 1:length(sizes)])
        @argcheck length(p) == num_params(length(x), sizes)
        state = x
        start_param = 1
        for (outsize, fun) in zip(sizes, funs)
            insize = length(state)
            nparams_dense_layer = (insize + 1) * outsize
            # @show start_param insize length(p[start_param : start_param + nparams_dense_layer - 1])
            state = dense(state, p[start_param : start_param + nparams_dense_layer - 1], insize, outsize, fun)
            start_param += nparams_dense_layer
        end
        return state
    end

    u0 = [0.0, 1.0]
    tspan = (0, 1)
    nstates=2

    nodes_per_element = 2
    time_supports = 10
    param_supports = 2

    # layer_sizes = (1,)
    # activations = (tanh,)
    layer_sizes = (8, 8, 1)
    activations = (tanh, tanh, identity)
    nparams=num_params(nstates, layer_sizes)

    optimizer = optimizer_with_attributes(Ipopt.Optimizer)
    model = InfiniteModel(optimizer)
    method = OrthogonalCollocation(nodes_per_element)

    @infinite_parameter(
        model, t in [tspan[1], tspan[2]], num_supports = time_supports, derivative_method = method
    )
    # @infinite_parameter(model, p[1:nparams] in [-1, 1], independent = true, num_supports = param_supports)

    @variable(model, -1 <= p[i = 1:nparams] <= 1)

    # state variables
    @variables(
        model,
        begin
            x[1:2], Infinite(t)
        end
    )

    @constraint(model, [i = 1:2], x[i](0) == u0[i])

    # NOTE: JUMP does not support vector valued functions
    # nlfun(x, p) = chain(x, p, layer_sizes, activations)
    # nlfun(x, p)
    # @register(model, nlfun(x, p))

    # https://jump.dev/JuMP.jl/stable/manual/nlp/#User-defined-functions-with-vector-inputs
    function scalar_fun(z...)
        # @show z typeof(z) z[1:nstates] z[nstates+1:end]
        x = collect(z[1:nstates])
        p = collect(z[nstates+1:end])
        chain(x, p, layer_sizes, activations)
    end

    scalar_fun(vcat(x, p)...)
    # @register scalar_fun(vcat(x, p)...)

    @constraints(
        model,
        begin
            ∂(x[1], t) == (1 - x[2]^2) * x[1] - x[2] + scalar_fun(vcat(x, p)...)[1]
            ∂(x[2], t) == x[1]
        end
    )

    @objective(model, Min, integral(x[1]^2 + x[2]^2 + scalar_fun(vcat(x, p)...)[1]^2, t))

    optimize!(model)

    jump_model = optimizer_model(model)

    solution_summary(jump_model; verbose=false)
end  #
