function run_simulation(
    controller,
    prob,
    params,
    tsteps;
    noise::@optional(Real)=nothing,
    vars::@optional(AbstractArray{<:Integer})=nothing,
    callback::@optional(DECallback)=nothing,
)
    if !isnothing(noise) && !isnothing(vars)
        @argcheck noise > zero(noise)
        @argcheck all(var in eachindex(prob.u0) for var in vars)

        function noiser(u, t, integrator)
            for var in vars
                u[var] += noise
            end
        end
        callback = FunctionCallingCallback(
            noiser;
            # funcat=tsteps,
            func_everystep=true,
        )
    end

    # integrate with given parameters
    solution = solve(prob, Tsit5(); p=params, saveat=tsteps, callback=callback)

    # construct arrays with the same type used by the integrator
    elements_type = eltype(solution.t)
    states = Array(solution)
    total_steps = size(states, 2)
    # state_dimension = size(states, 1)
    control_dimension = length(controller(solution.u[1], params))

    # regenerate controls from controller
    controls = zeros(elements_type, control_dimension, total_steps)
    for (step, state) in enumerate(solution.u)
        controls[:, step] = controller(state, params)
    end
    return solution.t, states, controls
end

function store_simulation(
    filename::Union{Nothing,String},
    controller::FastChain,
    prob::ODEProblem,
    params::AbstractVector{<:Real},
    tsteps::AbstractVector{<:Real};
    metadata=nothing::Union{Nothing,Dict},
    datadir=nothing::Union{Nothing,String},
    store_policy=true::Bool,
)
    if isnothing(datadir) || isnothing(filename)
        @info "Results not stored due to missing filename/datadir." maxlog = 1
        return nothing
    end

    if store_policy
        bson_path = joinpath(datadir, filename * ".bson")
        BSON.@save bson_path controller

        # weights_path = joinpath(datadir, filename * "_nnweights.csv")
        # CSV.write(weights_path, Tables.table(initial_params(controller)), writeheader=false)
    end

    times, states, controls = run_simulation(controller, prob, params, tsteps)

    state_headers = ["x$i" for i in 1:size(states, 1)]
    control_headers = ["c$i" for i in 1:size(controls, 1)]

    full_data = table(
        hcat(times, states', controls'); header=vcat(["t"], state_headers, control_headers)
    )

    CSV.write(joinpath(datadir, filename * ".csv"), full_data)

    if !isnothing(metadata)
        open(joinpath(datadir, filename * "_meta.json"), "w") do f
            JSON3.pretty(f, JSON3.write(metadata))
            println(f)
        end
    end
end
