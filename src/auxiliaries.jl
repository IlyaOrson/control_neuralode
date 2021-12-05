string_datetime() = replace(string(now()), (":" => "_"))

function generate_data_subdir(callerfile; current_datetime=nothing)
    parent = dirname(@__DIR__)
    isnothing(current_datetime) && (current_datetime = string_datetime())
    datadir = joinpath(parent, "data", basename(callerfile), current_datetime)
    @info "Generating data directory" datadir
    mkpath(datadir)
    return datadir
end

function local_grid(npoints, percentage; scale=1.0f0, type=:centered)
    @argcheck type in (:centered, :negative, :positive)
    @argcheck !iszero(scale)
    width = percentage * scale * npoints
    if type == :centered
        translation = width / 2.0f0
    elseif type == :negative
        translation = width * -1.0f0
    elseif type == :positive
        translation = 0.0f0
    end
    return [n * percentage * scale - translation for n in 0:npoints-1]
end

function controller_shape(controller)
    # this method is brittle as any function inside the Chain
    # will not be identified, could be a problem if those change dimensions

    # Flux Layers have fields (:weight, :bias, :σ)
    # FastLayers have fields (:out, :in, :σ, :initial_params, :bias)
    dims_input = [l.in for l in controller.layers[1:end] if typeof(l) <: FastDense]
    dims_output = [l.out for l in controller.layers[1:end] if typeof(l) <: FastDense]
    return push!(dims_input, pop!(dims_output))
end

function interpolant(timepoints, values; undersample=length(timepoints)÷4::Integer)
    @argcheck length(timepoints) == length(values)
    @argcheck undersample <= length(timepoints)
    space = Chebyshev(timepoints[1] .. timepoints[end])
    # http://juliaapproximation.github.io/ApproxFun.jl/stable/faq/
    # Create a Vandermonde matrix by evaluating the basis at the grid
    V = Array{Float64}(undef, length(timepoints), undersample)
    for k in 1:undersample
        V[:, k] = Fun(space, [zeros(k - 1); 1]).(timepoints)
    end
    return Fun(space, V \ vec(values))  # interpolant as one-variable function
end

# Feller, C., & Ebenbauer, C. (2014).
# Continuous-time linear MPC algorithms based on relaxed logarithmic barrier functions.
# IFAC Proceedings Volumes, 47(3), 2481–2488.
# https://doi.org/10.3182/20140824-6-ZA-1003.01022

function quadratic_relaxation(z, δ)
    return one(typeof(z)) / 2 * (((z - 2δ) / δ)^2 - one(typeof(z))) - log(δ)
end
exponential_relaxation(z, δ) = exp(one(typeof(z)) - z / δ) - one(typeof(z)) - log(δ)
function relaxed_log_barrier(z; δ=0.3f0)
    return max(z > δ ? -log(z) : exponential_relaxation(z, δ), zero(typeof(z)))
end
function relaxed_log_barrier(z, lower, upper; δ=(upper - lower) / convert(typeof(z), 2))
    return relaxed_log_barrier(z - lower; δ) + relaxed_log_barrier(upper - z; δ)
end

indicator_function(z; δ) = min(z, zero(typeof(z)))
