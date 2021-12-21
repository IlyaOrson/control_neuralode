const mpl = matplotlib

# const Line2D = matplotlib.lines.Line2D
# const Patch = matplotlib.patches.Patch

# const GridSpec = mpl.gridspec.GridSpec

plt.style.use("seaborn-colorblind")  # "ggplot"
palette = plt.cm.Dark2.colors

font = Dict(:family => "STIXGeneral", :size => 16)
savefig = Dict(:dpi => 600, :bbox => "tight")
lines = Dict(:linewidth => 4)
figure = Dict(:figsize => (8, 5))
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

function plot_collocation(controls_collocation, interpol, tsteps)
    plt.figure()
    finer_tsteps = range(tsteps[1], tsteps[end]; length=1000)
    plt.plot(finer_tsteps, [interpol(t) for t in finer_tsteps]; label="interpolation")
    plt.plot(tsteps, controls_collocation, "xg"; label="collocation")
    plt.title("Control collocation")
    plt.xlabel("time")
    plt.legend()
    plt.show()
end


abstract type PhasePlotMarker end

@kwdef struct IntegrationPath <: PhasePlotMarker
    points
    fmt = "m:"
    label = "Integration path"
    markersize = nothing
    linewidth = 4
end

@kwdef struct InitialState <: PhasePlotMarker
    points
    fmt = "bD"
    label = "Initial state"
    markersize = 12
    linewidth = nothing
end

@kwdef struct FinalState <: PhasePlotMarker
    points
    fmt = "r*"
    label = "Final state"
    markersize = 18
    linewidth = nothing
end

@kwdef struct ShadeConf
    indicator::Function
    cmap = "gray"
    transparency = 1
end

@kwdef struct ConstRef
    val::Real
    direction::Symbol
    class::Symbol
    var::Integer
    linestyle = "--"
    color = "r"
    label = "Constraint"
    linewidth = 3
    min = 0
    max = 1
    function ConstRef(
        val, direction, class, var, linestyle, color, label, linewidth, min, max
    )
        @argcheck direction in (:vertical, :horizontal)
        @argcheck class in (:state, :control)
        @argcheck 0 <= min < max <= 1
        return new(val, direction, class, var, linestyle, color, label, linewidth, min, max)
    end
end

@kwdef struct FuncRef
    fn::Function
    dom::Symbol
    class::Symbol
    var::Integer
    linestyle = "--"
    color = "r"
    label = "Constraint"
    linewidth = 3
    function FuncRef(fn, dom, class, var, linestyle, color, label, linewidth)
        @argcheck dom in (:space, :time)
        @argcheck class in (:state, :control)
        return new(fn, dom, class, var, linestyle, color, label, linewidth)
    end
end

function phase_portrait(
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
    @argcheck length(projection) == 2
    @argcheck all(x -> isa(x, Tuple) && length(x) == 2, coord_lims)
    @argcheck all(marker isa PhasePlotMarker for marker in markers)

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

function transparecy_scaler_abs(noise, perturbations; top=1, low=0.2)
    highest = maximum(abs, perturbations)
    amplitude = abs(noise) / highest
    return top - (top - low) * amplitude
end

function set_state_control_subplots(
    num_states, num_controls; annotation=nothing, refs=nothing
)
    if !isnothing(refs)
        @argcheck all(ref isa ConstRef for ref in refs)
        @argcheck all(
            ifelse(ref.class == :state, ref.var in 1:num_states, ref.var in 1:num_controls)
            for ref in refs
        ) BoundsError
    end

    fig_states, axs_states = plt.subplots(
        num_states;
        sharex="col",
        squeeze=false,
        # constrained_layout=true,  # incompatible with subplots_adjust
    )
    fig_controls, axs_controls = plt.subplots(
        num_controls;
        sharex="col",
        squeeze=false,
        # constrained_layout=true,  # incompatible with subplots_adjust
    )
    axs_states[1].set_title("States")
    axs_controls[1].set_title("Controls")
    axs_states[end].set_xlabel("time")
    axs_controls[end].set_xlabel("time")

    for s in 1:num_states
        axs_states[s].set_ylabel("state[$s]")
    end
    for c in 1:num_controls
        axs_controls[c].set_ylabel("control[$c]")
    end
    if !isnothing(refs)
        for r in refs
            ax = r.class == :state ? axs_states[r.var] : axs_controls[r.var]
            if r.direction == :vertical
                ax.axvline(
                    r.val;
                    ymin=r.min,
                    ymax=r.max,
                    label=r.label,
                    linestyle=r.linestyle,
                    color=r.color,
                )
            elseif r.direction == :horizontal
                ax.axhline(
                    r.val;
                    xmin=r.min,
                    xmax=r.max,
                    label=r.label,
                    linestyle=r.linestyle,
                    color=r.color,
                )
            end
        end
    end

    if !isnothing(annotation)
        fig_states.text(0, 0, string(annotation))
        fig_controls.text(0, 0, string(annotation))
    end

    return fig_states, axs_states, fig_controls, axs_controls
end

function plot_initial_perturbations(
    controller, prob, θ, tsteps, u0, specs; refs=nothing, funcs=nothing
)
    prob = remake(prob; p=θ)
    state_size = size(u0, 1)
    control_size = size(controller(u0, θ), 1)

    for spec in specs
        perturbations = local_grid(
            spec.samples, spec.percentage; scale=spec.scale, type=spec.type
        )
        cmap = ColorMap("tab10")
        fig_states, axs_states, fig_controls, axs_controls = set_state_control_subplots(
            length(u0), length(controller(u0, θ)); annotation=spec, refs
        )

        for noise in perturbations
            noise_vec = zeros(typeof(noise), length(u0))
            noise_vec[spec.variable] = noise

            # local prob = ODEProblem(dudt!, u0 + noise_vec, tspan, θ)
            perturbed_u0 = u0 + noise_vec
            prob = remake(prob; u0=perturbed_u0)

            times, states, controls = run_simulation(controller, prob, θ, tsteps)

            for s in 1:state_size
                axs_states[s].plot(
                    times,
                    states[s, :];
                    label="u0[$(spec.variable)] + " * format(noise; precision=2),
                    alpha=transparecy_scaler_abs(noise, perturbations),
                    c=cmap(s),
                )
            end
            for c in 1:control_size
                axs_controls[c].plot(
                    times,
                    controls[c, :];
                    label="u0[$(spec.variable)] + " * format(noise; precision=2),
                    alpha=transparecy_scaler_abs(noise, perturbations),
                    c=cmap(c + size(states, 1)),
                )
            end
        end
        legend_elements = [
            matplotlib.patches.Patch(;
                facecolor="black",
                edgecolor="black",
                label=format(noise; precision=2),
                alpha=transparecy_scaler_abs(noise, perturbations),
            ) for noise in sort(perturbations)
        ]
        fig_legend_div = 0.8
        fig_states.subplots_adjust(; right=fig_legend_div)
        legend_states = fig_states.legend(;
            handles=legend_elements,
            bbox_to_anchor=(fig_legend_div, 0.5),
            loc="center left",
            # borderaxespad=0,
            title="Perturbation",
        )
        fig_controls.subplots_adjust(; right=fig_legend_div)
        legend_controls = fig_controls.legend(;
            handles=legend_elements,
            bbox_to_anchor=(fig_legend_div, 0.5),
            loc="center left",
            # borderaxespad=0,
            title="Perturbation",
        )

        fig_states.show()
        fig_controls.show()

        # tight_layout alternative that considers the legend (or other artists)
        # bbox_extra_artists must be an iterable
        # fig.savefig("states_noise", bbox_extra_artists=(legend_states,), bbox_inches="tight")
        # fig.savefig("controls_noise", bbox_extra_artists=(legend_controls,), bbox_inches="tight")
    end
end

function plot_initial_perturbations_collocation(
    controller,
    prob,
    θ,
    tsteps,
    u0,
    specs,
    collocation::Function;
    refs=nothing,
    funcs=nothing,
    storedir=nothing,
)
    prob = remake(prob; p=θ)
    state_size = size(u0, 1)
    control_size = size(controller(u0, θ), 1)

    for spec in specs
        perturbations = local_grid(
            spec.samples, spec.percentage; scale=spec.scale, type=spec.type
        )
        for (tag, noise) in enumerate(perturbations)
            cmap = ColorMap("tab20")
            fig_states, axs_states, fig_controls, axs_controls = set_state_control_subplots(
                length(u0), length(controller(u0, θ)); annotation=spec, refs
            )

            noise_vec = zeros(typeof(noise), length(u0))
            noise_vec[spec.variable] = noise

            # local prob = ODEProblem(dudt!, u0 + noise_vec, tspan, θ)
            perturbed_u0 = u0 + noise_vec
            prob = remake(prob; u0=perturbed_u0)

            @info "simulation"
            @time times, states, controls = run_simulation(controller, prob, θ, tsteps)
            @info "collocation"
            @time infopt_model, states_collocation, controls_collocation = collocation(
                perturbed_u0; state_constraints=true
            )
            @info "interpolation"
            @time interpol = interpolant(tsteps, controls_collocation)

            for s in 1:state_size
                axs_states[s].plot(
                    times,
                    states[s, :];
                    label="policy",
                    # alpha=transparecy_scaler_abs(noise, perturbations),
                    color=cmap(2s - 2),
                )
                axs_states[s].plot(
                    times,
                    states_collocation[s, :];
                    label="collocation",
                    marker="x",
                    linestyle="None",
                    # alpha=transparecy_scaler_abs(noise, perturbations),
                    color=cmap(2s - 1),
                )
            end
            for c in 1:control_size
                axs_controls[c].plot(
                    times,
                    controls[c, :];
                    label="policy",
                    # alpha=transparecy_scaler_abs(noise, perturbations),
                    color=cmap(2c - 2 + 2size(states, 1)),
                )
                axs_controls[c].plot(
                    times,
                    controls_collocation[c, :];
                    label="collocation",
                    marker="x",
                    linestyle="None",
                    # alpha=transparecy_scaler_abs(noise, perturbations),
                    color=cmap(2c - 1 + 2size(states, 1)),
                )
            end
            title = "u0[$(spec.variable)] + " * format(noise; precision=2)
            fig_states.suptitle(title)
            fig_controls.suptitle(title)

            legend_elements = [
                matplotlib.lines.Line2D(
                    [0], [0]; color="black", linewidth=3, label="policy"
                ),
                matplotlib.lines.Line2D(
                    [0],
                    [0];
                    color="black",
                    marker="x",
                    linestyle="None",
                    linewidth=3,
                    label="collocation",
                ),
            ]

            # fig_legend_div = 0.8
            # fig_states.subplots_adjust(; right=fig_legend_div)
            legend_states = fig_states.legend(;
                handles=legend_elements,
                # bbox_to_anchor=(fig_legend_div, 0.5),
                # loc="center left",
            )
            # fig_controls.subplots_adjust(; right=fig_legend_div)
            legend_controls = fig_controls.legend(;
                handles=legend_elements,
                # bbox_to_anchor=(fig_legend_div, 0.5),
                # loc="center left",
            )

            if !isnothing(storedir)
                # tight_layout alternative that considers the legend (or other artists)
                # bbox_extra_artists must be an iterable
                filename = "u0_$(spec.variable)_" * sprintf1("%04d", tag)
                fig_states.savefig(
                    joinpath(storedir, filename * "_states.png");
                    bbox_extra_artists=(legend_states,),
                    # bbox_inches="tight",  # this adjusts filesize slighly... problematic for gifs
                )
                fig_controls.savefig(
                    joinpath(storedir, filename * "_controls.png");
                    bbox_extra_artists=(legend_controls,),
                    # bbox_inches="tight",
                )
            else
                fig_states.show()
                fig_controls.show()
            end
        end
    end
end
