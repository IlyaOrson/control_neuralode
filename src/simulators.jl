function run_simulation(
    controlODE::ControlODE,
    params;
    control_input=:state,
    noise::Union{Nothing,Real}=nothing,
    vars::Union{Nothing,AbstractArray{<:Integer}}=nothing,
    callback::Union{Nothing,DECallback}=nothing,
    kwargs...,
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

    # regenerate controls from controlODE.controller
    if control_input == :state
        control_dimension = length(controlODE.controller(solution.u[begin], params))
        controls = zeros(elements_type, control_dimension, total_steps)
        for (step, state) in enumerate(solution.u)
            controls[:, step] = controlODE.controller(state, params)
        end
    elseif control_input == :time
        control_dimension = length(controlODE.controller(solution.t[begin], params))
        controls = zeros(elements_type, control_dimension, total_steps)
        for (step, time) in enumerate(solution.t)
            controls[:, step] = controlODE.controller(time, params)
        end
    else
        @check control_input in [:state, :time]
    end

    return solution.t, states, controls
end

# TODO: use DrWatson jl
function store_simulation(
    filename::Union{Nothing,String},
    controlODE::ControlODE,
    params::AbstractVector{<:Real};
    metadata::Union{Nothing,Dict}=nothing,
    datadir::Union{Nothing,String}=nothing,
)
    if isnothing(datadir) || isnothing(filename)
        @info "Results not stored due to missing filename/datadir." maxlog = 1
        return nothing
    end

    params_path = joinpath(datadir, filename * "_params.jls")
    serialize(params_path, params)

    # weights_path = joinpath(datadir, filename * "_nnweights.csv")
    # CSV.write(weights_path, Tables.table(params), writeheader=false)

    times, states, controls = run_simulation(controlODE, params)

    state_headers = ["x$i" for i in 1:size(states, 1)]
    control_headers = ["c$i" for i in 1:size(controls, 1)]

    header = vcat(["t"], state_headers, control_headers)
    data = hcat(times, permutedims(states), permutedims(controls))

    filepath_csv = joinpath(datadir, filename * ".csv")

    # data_table = table(data; header)
    # CSV.write(filepath_csv, data_table)

    open(filepath_csv, "w") do io
        writedlm(io, vcat(permutedims(header), data), ',')
    end

    if !isnothing(metadata)
        open(joinpath(datadir, filename * "_meta.json"), "w") do f
            JSON3.pretty(f, metadata)
        end
    end
end
