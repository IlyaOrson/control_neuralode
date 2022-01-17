# https://discourse.julialang.org/t/aliases-for-union-t-nothing-and-union-t-missing/15402/41
macro optional(ex)
    return :(Union{$ex,Nothing})
end

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

function controller_shape(controller)
    # this method is brittle as any function inside the Chain
    # will not be identified, could be a problem if those change dimensions

    # Flux Layers have fields (:weight, :bias, :σ)
    # FastLayers have fields (:out, :in, :σ, :initial_params, :bias)
    dims_input = [l.in for l in controller.layers[1:end] if typeof(l) <: FastDense]
    dims_output = [l.out for l in controller.layers[1:end] if typeof(l) <: FastDense]
    return push!(dims_input, pop!(dims_output))
end

function scaled_sigmoids(control_ranges)
    return (x, p) -> [
        (control_ranges[i][2] - control_ranges[i][1]) * sigmoid_fast(x[i]) +
        control_ranges[i][1] for i in eachindex(control_ranges)
    ]
end

find_array_param(arr::AbstractArray{T}) where {T} = T

struct ControlODE{uType<:Real,tType<:Real}
    controller::Function
    system!::Function
    u0::AbstractVector{uType}
    tspan::Tuple{tType,tType}
    tsteps::AbstractVector{tType}
    integrator::AbstractODEAlgorithm
    sensealg::AbstractSensitivityAlgorithm
    prob::AbstractODEProblem
    function ControlODE(
        controller,
        system!,
        u0,
        tspan;
        tsteps=range(tspan...; length=101),
        Δt::@optional(Real)=nothing,
        integrator=INTEGRATOR,  # Tsit5()
        sensealg=SENSEALG,
    )
        # check tsteps construction
        @argcheck isnothing(tsteps) || isnothing(Δt)
        if !isnothing(Δt)
            tsteps = range(tspan...; step=Δt)
        else
            @argcheck tspan[begin] == tsteps[begin]
            @argcheck tspan[end] == tsteps[end]
        end
        # check domain types
        time_type = find_array_param(tsteps)
        space_type = find_array_param(u0)
        control_type = find_array_param(controller(u0, initial_params(controller)))
        @argcheck space_type == control_type

        # cosntruct ODE problem
        dudt!(du, u, p, t) = system!(du, u, p, t, controller)
        prob = ODEProblem(dudt!, u0, tspan)

        return new{space_type,time_type}(
            controller, system!, u0, tspan, tsteps, integrator, sensealg, prob
        )
    end
end
# TODO follow recommended interface https://github.com/SciML/CommonSolve.jl
function solve(code::ControlODE, params; kwargs...)
    return solve(
        code.prob, code.integrator; p=params, code.tsteps, code.sensealg, kwargs...
    )
end

function ChevyshevInterpolation(
    timepoints, values; undersample=length(timepoints) ÷ 4::Integer
)
    @argcheck length(timepoints) == length(values)
    @argcheck undersample <= length(timepoints)
    space = Chebyshev(Interval(timepoints[1], timepoints[end]))
    # http://juliaapproximation.github.io/ApproxFun.jl/stable/faq/
    # Create a Vandermonde matrix by evaluating the basis at the grid
    V = Array{Float64}(undef, length(timepoints), undersample)
    for k in 1:undersample
        V[:, k] = Fun(space, [zeros(k - 1); 1]).(timepoints)
    end
    return Fun(space, V \ vec(values))  # ChevyshevInterpolation as one-variable function
end

function collocation_preconditioner(u0, collocation; plot=true, kwargs...)
    @info "Preconditioning with collocation..."
    infopt_model, time_collocation, states_collocation, controls_collocation = collocation(
        u0; kwargs...
    )
    num_controls = size(controls_collocation, 1)
    # interpolations = [
    #     DataInterpolations.LinearInterpolation(controls_collocation[i, :], time_collocation) for i in 1:num_controls
    # ]
    interpolations = [
        ChevyshevInterpolation(time_collocation, controls_collocation[i, :]) for
        i in 1:num_controls
    ]
    function control_profile(t, p)
        Zygote.ignore() do
            return [interpolations[i](t) for i in 1:num_controls]
        end
    end
    if plot
        for c in 1:num_controls
            display(
                lineplot(
                    t -> control_profile(t, nothing)[c],
                    time_collocation[begin],
                    time_collocation[end];
                    xlim=(time_collocation[begin], time_collocation[end]),
                ),
            )
            plot_collocation(
                controls_collocation[c, :], interpolations[c], time_collocation
            )
        end
    end
    return control_profile,
    infopt_model, time_collocation, states_collocation,
    controls_collocation
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
