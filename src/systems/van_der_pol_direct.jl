using InfiniteOpt, ArgCheck, Ipopt
import ControlNeuralODE as cnode

function van_der_pol_direct(; store_results=false::Bool)

    u0 = [0.0, 1.0]
    tspan = (0, 1)
    nstates = length(u0)

    nodes_per_element = 2
    time_supports = 10

    # layer_sizes = (1,)
    # activations = (tanh,)
    layer_sizes = (8, 8, 1)
    activations = (tanh, tanh, identity)

    nparams = cnode.count_params(nstates, layer_sizes)
    xavier_weights = cnode.start_values_sampler(nstates, layer_sizes)

    function vector_fun(z)
        # @show z typeof(z) z[1:nstates] z[nstates+1:end]
        x = collect(z[1:nstates])
        p = collect(z[(nstates + 1):end])
        return cnode.chain(x, p, layer_sizes, activations)
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

        optimize!(model)  # TODO: use optimize_infopt!() instead

        jump_model = optimizer_model(model)

        solution_summary(jump_model; verbose=false)
        return model
    end

    model_opt = infopt_direct()
    result = cnode.extract_infopt_results(model_opt)

    # TODO: deduplicate this
    function system!(du, u, p, t, controller; input=:state)
        @argcheck input in (:state, :time)

        # neural network outputs the controls taken by the system
        x1, x2 = u

        if input == :state
            c1 = controller(u, p)[1]  # control based on state and parameters
        elseif input == :time
            c1 = controller(t, p)[1]  # control based on time and parameters
        end

        # dynamics of the controlled system
        x1_prime = (1 - x2^2) * x1 - x2 + c1
        x2_prime = x1
        # x3_prime = x1^2 + x2^2 + c1^2

        # update in-place
        @inbounds begin
            du[1] = x1_prime
            du[2] = x2_prime
            # du[3] = x3_prime
        end
    end

    controller = (x, p) -> cnode.chain(x, p, layer_sizes, activations)
    controlODE = cnode.ControlODE(controller, system!, u0, tspan; Δt = 0.1f0, params=result.params)
    # @infiltrate
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
van_der_pol_direct()
