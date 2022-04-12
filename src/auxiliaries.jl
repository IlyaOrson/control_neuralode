find_array_param(arr::AbstractArray{T}) where {T} = T

string_datetime() = replace(string(now()), (":" => "_"))

function generate_data_subdir(
    callerfile; parent=dirname(@__DIR__), subdir=string_datetime()
)
    datadir = joinpath(parent, "data", basename(callerfile), subdir)
    @info "Generating data directory" datadir
    mkpath(datadir)
    return datadir
end

function local_grid(npoints::Integer, percentage::Real; scale=1.0f0, type=:centered)
    @argcheck zero(percentage) < percentage <= one(percentage)
    @argcheck type in (:centered, :negative, :positive)
    @argcheck !iszero(scale)
    width = percentage * scale * npoints
    if type == :centered
        translation = width / 2.0f0
    elseif type == :negative
        translation = width
    elseif type == :positive
        translation = 0.0f0
    end
    return [n * percentage * scale - translation for n in 0:(npoints - 1)]
end

function square_bounds(u0, arista)
    low_bounds = u0 .- repeat([arista / 2], length(u0))
    high_bounds = u0 .+ repeat([arista / 2], length(u0))
    bounds = [(l, h) for (l, h) in zip(low_bounds, high_bounds)]
    return bounds
end

function controller_shape(controller)
    # this method is brittle as any function inside the Chain
    # will not be identified, could be a problem if those change dimensions

    # Flux Layers have fields (:weight, :bias, :σ)
    # FastLayers have fields (:out, :in, :σ, :initial_params, :bias)
    dims_input = [l.in for l in controller.layers[1:end] if typeof(l) <: FastLayer]
    dims_output = [l.out for l in controller.layers[1:end] if typeof(l) <: FastLayer]
    return push!(dims_input, pop!(dims_output))
end

function compose_layers(controller::FastChain, params, u0, L)
    @argcheck 1 <= L <= length(controller.layers)
    @argcheck length(initial_params(controller)) == length(params)
    index = 1
    x = u0
    for i in 1:L
        layer = controller.layers[i]
        if typeof(layer) <: FastLayer
            ip = initial_params(layer)
            s = length(ip)
            p = params[index:(index + s - 1)]
            index += s
            x = layer(x, p)
        else
            x = layer(x, params)
        end
    end
    return x
end

function scaled_sigmoids(control_ranges)
    return (x, p) -> [
        (control_ranges[i][2] - control_ranges[i][1]) * sigmoid_fast(x[i]) +
        control_ranges[i][1] for i in eachindex(control_ranges)
    ]
end

function optimize_infopt!(infopt_model::InfiniteModel)
    optimize!(infopt_model)

    # list possible termination status: model |> termination_status |> typeof
    jump_model = optimizer_model(infopt_model)
    # OPTIMAL = 1, LOCALLY_SOLVED = 4
    if Int(termination_status(jump_model)) ∉ (1, 4)
        @error raw_status(jump_model) termination_status(jump_model)
        error("The collocation optimization failed.")
    else
        @info solution_summary(jump_model; verbose=false)
        # @info "Objective value" objective_value(infopt_model)
    end
    return infopt_model
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
