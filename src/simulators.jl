function run_simulation(
    controlODE,
    params;
    noise::Union{Nothing, Real}=nothing,
    vars::Union{Nothing, AbstractArray{<:Integer}}=nothing,
    callback::Union{Nothing, DECallback}=nothing,
    kwargs...
)
    if !isnothing(noise) && !isnothing(vars)
        if !isnothing(callback)
            @warn "Supplied callback will be replaced by a noise callback."
        end
        @argcheck noise > zero(noise)
        @argcheck all(var in eachindex(controlODE.u0) for var in vars)

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
    solution = solve(controlODE, params; callback, kwargs...)

    # construct arrays with the same type used by the integrator
    elements_type = eltype(solution.t)
    states = Array(solution)
    total_steps = size(states, 2)
    # state_dimension = size(states, 1)
    control_dimension = length(controlODE.controller(solution.u[1], params))

    # regenerate controls from controlODE.controller
    controls = zeros(elements_type, control_dimension, total_steps)
    for (step, state) in enumerate(solution.u)
        controls[:, step] = controlODE.controller(state, params)
    end
    return solution.t, states, controls
end

# TODO: use DrWatson jl
function store_simulation(
    filename::Union{Nothing,String},
    controlODE::ControlODE,
    params::AbstractVector{<:Real};
    metadata=nothing::Union{Nothing,Dict},
    datadir=nothing::Union{Nothing,String},
    store_policy=true::Bool,
)
    if isnothing(datadir) || isnothing(filename)
        @info "Results not stored due to missing filename/datadir." maxlog = 1
        return nothing
    end

    if store_policy
        policy_path = joinpath(datadir, filename * ".jls")
        open(io -> serialize(io, controlODE.controller), policy_path, "w")

        # weights_path = joinpath(datadir, filename * "_nnweights.csv")
        # CSV.write(weights_path, Tables.table(initial_params(controlODE.controller)), writeheader=false)
    end

    times, states, controls = run_simulation(controlODE, params)

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
