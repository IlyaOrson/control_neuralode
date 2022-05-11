function chebyshev_interpolation(
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
    return Fun(space, V \ vec(values))  # chebyshev_interpolation as one-variable function
end

function interpolant_controller(collocation; plot=nothing)

    num_controls = size(collocation.controls, 1)

    interpolations = [
        LinearInterpolation(collocation.controls[i, :], collocation.times) for i in 1:num_controls
        # CubicSpline(collocation.controls[i, :], collocation.times) for i in 1:num_controls
    ]
    # interpolations = [
    #     chebyshev_interpolation(collocation.times, collocation.controls[i, :]) for
    #     i in 1:num_controls
    # ]

    function control_profile(t, p)
        Zygote.ignore() do
            return [interpolations[i](t) for i in 1:num_controls]
        end
    end

    if !isnothing(plot)
        @argcheck plot in [:unicode, :pyplot]
        for c in 1:num_controls
            if plot == :unicode
                display(
                    lineplot(
                        t -> control_profile(t, nothing)[c],
                        collocation.times[begin+1],
                        collocation.times[end-1];
                        xlim=(collocation.times[begin], collocation.times[end]),
                    ),
                )
            else
                plot_collocation(
                    collocation.controls[c, begin+1:end-1], interpolations[c], collocation.times
                )
            end
        end
    end
    return control_profile
end
