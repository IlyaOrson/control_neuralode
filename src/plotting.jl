const mpl = matplotlib
# const GridSpec = mpl.gridspec.GridSpec

plt.style.use("seaborn-colorblind")  # "ggplot"
palette = plt.cm.Dark2.colors

font = Dict(:family => "STIXGeneral", :size => 16)
savefig = Dict(:dpi => 600, :bbox => "tight")
lines = Dict(:linewidth => 4)
figure = Dict(:figsize => (8, 4))
axes = Dict(:prop_cycle => mpl.cycler(; color=palette))
legend = Dict(:fontsize => "x-large")  # medium for presentations, x-large for papers

mpl.rc("font"; font...)
mpl.rc("savefig"; savefig...)
mpl.rc("lines"; lines...)
mpl.rc("figure"; figure...)
mpl.rc("axes"; axes...)
mpl.rc("legend"; legend...)

function fun_plotter(fun, array; xlim=(0, 0))
    output = map(fun, eachrow(array)...)
    return lineplot(output; title="Custom Function", name="fun", ylim=extrema(output), xlim)
end

# handy terminal plots
function unicode_plotter(states, controls; only=nothing, vars=nothing, fun=nothing)
    @assert size(states, 2) == size(controls, 2)
    xlim = (0, size(states, 2))
    if only == :states
        if !isnothing(fun)
            return fun_plotter(fun, states; xlim)
        end
        typeof(vars) <: Vector && (states = @view states[vars, :])
        ylim = states |> extrema
        tag = isnothing(vars) ? "x1" : "x$(vars[1])"
        plt = lineplot(
            states[1, :]; title="State Evolution", name=tag, xlabel="step", ylim, xlim
        )
        for (i, s) in enumerate(eachrow(states[2:end, :]))
            tag = isnothing(vars) ? "x$(i+1)" : "x$(vars[i+1])"
            lineplot!(plt, collect(s); name=tag)
        end
    elseif only == :controls
        if !isnothing(fun)
            return fun_plotter(fun, controls; xlim)
        end
        typeof(vars) <: Vector && (controls = @view controls[vars, :])
        ylim = controls |> extrema
        tag = isnothing(vars) ? "c1" : "c$(vars[1])"
        plt = lineplot(
            controls[1, :]; title="Control Evolution", name=tag, xlabel="step", ylim, xlim
        )
        for (i, s) in enumerate(eachrow(controls[2:end, :]))
            tag = isnothing(vars) ? "c$(i+1)" : "c$(vars[i+1])"
            lineplot!(plt, collect(s); name=tag)
        end
    else
        if !isnothing(fun)
            return fun_plotter(fun, hcat(states, controls); xlim)
        end
        ylim = Iterators.flatten((states, controls)) |> extrema
        plt = lineplot(
            states[1, :];
            title="State and Control Evolution",
            name="x1",
            xlabel="step",
            ylim,
            xlim,
        )
        for (i, s) in enumerate(eachrow(states[2:end, :]))
            lineplot!(plt, collect(s); name="x$(i+1)")
        end
        for (i, c) in enumerate(eachrow(controls))
            lineplot!(plt, collect(c); name="c$i")
        end
    end
    return plt
end

function plot_simulation(
    controller,
    prob,
    params,
    tsteps;
    show=nothing,
    only=nothing,
    vars=nothing,
    fun=nothing,
    yrefs=nothing,
)
    !isnothing(show) && @info show

    # TODO: use times in plotting?
    times, states, controls = run_simulation(controller, prob, params, tsteps)
    plt = unicode_plotter(states, controls; only, vars, fun)
    if !isnothing(yrefs)
        for yref in yrefs
            lineplot!(plt, x -> yref; name="$yref")
        end
    end
    display(plt)
    return false  # if return true, then optimization stops
end

@kwdef struct PlotConf
    points
    fmt = "."
    label = nothing
    markersize = nothing
    linewidth = nothing
end

@kwdef struct ShadeConf
    indicator::Function
    cmap = "gray"
    transparency = 1
end

function phase_plot(
    system!,
    controller,
    params,
    phase_time,
    coord_lims;  #xlims, ylims
    points_per_dim=1000,
    dimension=2,
    projection=[1, 2],
    markers=nothing,
    start_points=nothing,
    start_points_x=nothing,
    start_points_y=nothing,
    title=nothing,
    shader=nothing,
    kwargs...,
)
    @assert length(projection) == 2
    @assert all(x -> isa(x, Tuple) && length(x) == 2, coord_lims)

    function stream_interface(coords...)
        u = zeros(Float32, dimension)
        du = zeros(Float32, dimension)
        copyto!(u, coords)
        # du = deepcopy(coords)
        system!(du, u, params, phase_time, controller)
        return du
    end

    # evaluate system over each combination of coords in the specified ranges

    # NOTE: float64 is relevant for the conversion to pyplot due to inner
    #       numerical checks of equidistant input in the streamplot function
    ranges = [range(Float64.(lims)...; length=points_per_dim) for lims in coord_lims]
    xpoints, ypoints = collect.(ranges[projection])

    # NOTE: the transpose is required to get f.(a',b) instead of the default f.(a, b')
    xgrid, ygrid = ndgrid(xpoints, ypoints)  # NOTE: is this switched?
    phase_array_tuples = stream_interface.(xgrid, ygrid)'
    # phase_array_tuples = stream_interface.(xpoints', ypoints)

    xphase, yphase = [getindex.(phase_array_tuples, dim) for dim in projection]

    magnitude = map((x, y) -> sqrt(sum(x^2 + y^2)), xphase, yphase)

    fig = plt.figure()
    # gs = GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    if isnothing(start_points) && !isnothing(start_points_x) && !isnothing(start_points_y)
        @assert size(start_points_x) == size(start_points_y)
        start_grid_x, start_grid_y = ndgrid(start_points_x, start_points_y)
        start_points = hcat(reshape(start_grid_x, :, 1), reshape(start_grid_y, :, 1))
    end

    # integration_direction = isnothing(start_points) ? "both" : "forward"
    ax = fig.add_subplot()
    strm = ax.streamplot(
        xpoints,
        ypoints,
        xphase,
        yphase;
        color=magnitude,
        linewidth=1.5,
        density=1.5,
        cmap="summer",
        kwargs...,
    )
    if !isnothing(start_points)
        start_points = start_points[:, projection]
        ax.plot(start_points[:, 1], start_points[:, 2], "kX"; markersize=12)
        strm = ax.streamplot(
            xpoints,
            ypoints,
            xphase,
            yphase;
            color="darkorchid",
            linewidth=3,
            density=20,
            start_points,
            integration_direction="forward",
            kwargs...,
        )
    end

    # displaying points (handles multiple points as horizontally concatenated)
    if !isnothing(markers)
        for plotconf in markers
            points_projected = plotconf.points[projection, :]
            ax.plot(
                points_projected[1, :],
                points_projected[2, :],
                plotconf.fmt;
                label=plotconf.label,
                markersize=plotconf.markersize,
                linewidth=plotconf.linewidth,
            )
        end
    end

    xlims, ylims = coord_lims[projection]

    if !isnothing(shader)
        mask = shader.indicator.(xgrid, ygrid)'
        ax.imshow(
            mask;
            extent=(xlims..., ylims...),
            alpha=shader.transparency,
            cmap=shader.cmap,
            aspect="auto",
        )
    end

    ax.set(; xlim=xlims .+ (-.05, 0.05), ylim=ylims .+ (-.05, 0.05))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    !isnothing(title) && ax.set_title(title)

    fig.colorbar(strm.lines)
    ax.legend()

    # remove frame
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.spines["left"].set_visible(false)

    plt.tight_layout()

    return plt.show()
end
