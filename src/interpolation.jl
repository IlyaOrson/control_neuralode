function interpolant_controller(collocation; plot=nothing)

    num_controls = size(collocation.controls, 1)

    interpolations = [
        LinearInterpolation(collocation.controls[i, :], collocation.times) for i in 1:num_controls
        # CubicSpline(collocation.controls[i, :], collocation.times) for i in 1:num_controls
    ]

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
                        title="Collocation result",
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
