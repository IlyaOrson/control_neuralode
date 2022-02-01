using InfiniteOpt, ArgCheck, Ipopt
import ControlNeuralODE as cnode

function van_der_pol_direct(; store_results=false::Bool)

    # TODO: refactor
    function dense(x, p, out_size, fun=identity)
        in_size = length(x)
        @argcheck length(p) == (in_size + 1) * out_size
        matrix = reshape(p[1:(out_size * in_size)], out_size, in_size)
        biases = p[(out_size * in_size + 1):end]  # end = out_size * (in_size + 1)
        # @show size(matrix) size(biases)
        return fun.(matrix * x .+ biases)
    end

    function count_params(state_size, layers_sizes)
        in_size = state_size
        sum = 0
        for out_size in layers_sizes
            # @show in_size out_size
            sum += (in_size + 1) * out_size
            in_size = out_size
        end
        return sum
    end

    function chain(x, p, sizes, funs)
        @argcheck length(p) == count_params(length(x), sizes)
        state = x
        start_param = 1
        for (out_size, fun) in zip(sizes, funs)
            in_size = length(state)
            nparams_dense_layer = (in_size + 1) * out_size
            # @show start_param insize length(p[start_param : start_param + nparams_dense_layer - 1])
            state = dense(
                state, p[start_param:(start_param + nparams_dense_layer - 1)], out_size, fun
            )
            start_param += nparams_dense_layer
        end
        return state
    end

    function start_values_sampler(
        state_size, layers_sizes; factor=(in, out) -> 2 / (in + out)
    )
        in_size = state_size
        total_params = count_params(state_size, layers_sizes)
        sample_array = zeros(total_params)
        start_param = 1
        for out_size in layers_sizes
            nparams_dense_layer = (in_size + 1) * out_size
            # Xavier initialization: variance = 2/(in+out)
            # factor = 2/(in_size + out_size)
            samples = factor(in_size, out_size)^2 * randn(nparams_dense_layer)
            sample_array[start_param:(start_param + nparams_dense_layer - 1)] = samples
            start_param += nparams_dense_layer
            in_size = out_size
        end
        return sample_array
    end

    u0 = [0.0, 1.0]
    tspan = (0, 1)
    nstates = length(u0)

    nodes_per_element = 2
    time_supports = 10

    # layer_sizes = (1,)
    # activations = (tanh,)
    layer_sizes = (8, 8, 1)
    activations = (tanh, tanh, identity)

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

    function infopt_direct()
        optimizer = optimizer_with_attributes(Ipopt.Optimizer)
        model = InfiniteModel(optimizer)
        method = OrthogonalCollocation(nodes_per_element)

        @infinite_parameter(
            model,
            t in [tspan[1], tspan[2]],
            num_supports = time_supports,
            derivative_method = method
        )
        # @infinite_parameter(model, p[1:nparams] in [-1, 1], independent = true, num_supports = param_supports)
        # @infiltrate
        # TODO: remove hard constraints
        @variable(model, -10 <= p[i=1:nparams] <= 10, start = xavier_weights[i])

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

        optimize!(model)  # TODO: use optimize_collocation!() instead

        jump_model = optimizer_model(model)

        solution_summary(jump_model; verbose=false)
        return model
    end

    function extract_infopt_results(model; time=:t, state=:x, control=:c, param=:p)
        # TODO: could be more general
        @argcheck has_values(model)

        model_keys = keys(model.obj_dict)

        times = supports(model[time])
        states = hcat(value.(model[state])...) |> permutedims

        results = (; times=times, states=states)

        if control in model_keys
            controls = hcat(value.(model[control])...) |> permutedims
            results = merge(results, (; controls=controls))
        end
        if param in model_keys
            params = hcat(value.(model[param])...) |> permutedims
            results = merge(results, (; params=params))
        end
        return results
    end

    model_opt = infopt_direct()
    result = extract_infopt_results(model_opt)

    controller = (x, p) -> chain(x, p, layer_sizes, activations)
    controlODE = cnode.ControlODE(controller, system!, u0, tspan; Δt = 0.1f0, params=result.params)
    @infiltrate
    cnode.phase_portrait(
        controlODE,
        result.params,
        cnode.square_bounds(u0, 7);
        markers=[
            cnode.InitialMarkers(; points=result.states[:, 1]),
            cnode.IntegrationPath(; points=result.states),
            cnode.FinalMarkers(; points=result.states[:, end]),
        ],
    )
    return nothing
end
