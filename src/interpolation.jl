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

function interpolant_controller(collocation; plot=true)
    @info "Solving through collocation..."

    num_controls = size(collocation.controls, 1)

    # interpolations = [
    #     DataInterpolations.LinearInterpolation(collocation.controls[i, :], collocation.times) for i in 1:num_controls
    # ]
    interpolations = [
        chebyshev_interpolation(collocation.times, collocation.controls[i, :]) for
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
                    collocation.times[begin],
                    collocation.times[end];
                    xlim=(collocation.times[begin], collocation.times[end]),
                ),
            )
            plot_collocation(
                collocation.controls[c, :], interpolations[c], collocation.times
            )
        end
    end
    return control_profile
end
