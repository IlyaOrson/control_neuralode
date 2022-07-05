RESULTS_REGEX = r"delta_(\d+\.\d+)_iter_(\d+)"

function name_interpolation(delta, iter)
    return "delta_$(round(delta, digits=3))_iter_$iter"
end

function extract_name_values(filename; re=RESULTS_REGEX)
    matcher = match(re, filename)
    if isnothing(matcher)
        return nothing
    else
        delta_s, iter_s = matcher
        iter = parse(Int, iter_s)
        delta = parse(Float64, delta_s)
        return iter, delta
    end
end

function filter_matching_filepath(filepaths, re; extension=nothing)
    for filepath in filepaths
        relevant = !isnothing(extension) && endswith(filepath, extension)
        if relevant
            matcher = match(re, filepath)
            if !isnothing(matcher)
                # will only return the first match
                return filepath
            end
        end
    end
end

function extract_all_name_values(dirpath)
    iter_delta_dict = SortedDict{Int,Float64}()
    filepaths = readdir(dirpath; join=true, sort=true)

    for filepath in filepaths
        vals = extract_name_values(filepath)
        isnothing(vals) && continue
        iter, delta = vals
        iter_delta_dict[iter] = delta
    end
    return iter_delta_dict
end

function load_penalization_round(dirpath, system; iter, delta)  # system = BioReactor(), VanDerPol() ...
    filepaths = readdir(dirpath; join=true, sort=true)

    name = name_interpolation(delta, iter)
    filepath = filter_matching_filepath(filepaths, Regex(name); extension="params.jls")

    controlODE = ControlODE(system)
    weights = deserialize(filepath)

    return controlODE, weights
end

function plot_penalization_rounds(
    dirpath,
    system;
    state_var=nothing,
    control_var=nothing,
    timesteps=100,
    palette="Blues",  # Purples, Blues, Greens, Oranges, Reds
    running_ref=nothing,
    final_ref=nothing,
    ref_color="orange",
    transparency=0.7,
    saveto=nothing,
    colorant="delta",
    )
    @argcheck colorant in ["delta", "iter"]
    @argcheck !all(map(x -> isnothing(x), (state_var, control_var)))  "Please specify either state_var or control_var."
    @argcheck !all(map(x -> !isnothing(x), (state_var, control_var)))  "Please specify just one of state_var or control_var."

    iter_delta_dict = extract_all_name_values(dirpath)
    pprint(iter_delta_dict)
    iters = collect(keys(iter_delta_dict))
    deltas = collect(values(iter_delta_dict)) |> sort
    color_range = range(0.2, 1, length(iter_delta_dict))
    cmap = ColorMap(palette)
    cmap_range = ColorMap(palette)(color_range)
    fig, ax = plt.subplots()
    local times
    for (i, (iter, delta)) in enumerate(iter_delta_dict)
        controlODE, weights = load_penalization_round(
            dirpath, system; iter, delta
        )
        span = controlODE.tspan[end] - controlODE.tspan[begin]
        times, states, controls = run_simulation(controlODE, weights; saveat=span / timesteps)

        if !isnothing(state_var)
            ax.plot(
                times,
                states[state_var,:];
                color=cmap_range[i, :],
                label="δ=$delta",
            alpha=transparency,
            )
            plt.ylabel("x_$state_var")
        elseif !isnothing(control_var)
            ax.plot(
                times,
                controls[control_var,:];
                color=cmap_range[i, :],
                label="δ=$delta",
                alpha=transparency,
            )
            plt.ylabel("c_$control_var")
        end
    end
    final_time_gap = (times[end] - times[begin]) / 10
    !isnothing(running_ref) && plt.plot(
        (times[begin], times[end] - final_time_gap),
        (running_ref, running_ref);
        zorder=110,
        color=ref_color,
        alpha=transparency,
        ls="--",
    )
    !isnothing(final_ref) && plt.plot(
        (times[end] - final_time_gap, times[end]),
        (running_ref, running_ref);
        zorder=110,
        color=ref_color,
        alpha=transparency,
        ls="--",
    )
    # plt.legend(; fontsize="x-small", loc="center left", bbox_to_anchor=(1.02, 0.5))
    # @infiltrate
    local norm
    if colorant == "iter"
        norm = matplotlib.colors.Normalize(; vmin=iters[begin], vmax=iters[end])
    elseif colorant == "delta"
        norm = matplotlib.colors.Normalize(; vmin=deltas[end], vmax=deltas[begin])
    end
    fig.colorbar(
        matplotlib.cm.ScalarMappable(; norm=norm, cmap=cmap);
        # cax=ax,
        orientation="vertical",
        # label=L"\delta",
        label="iteration",
    )
    plt.xlabel("time")
    plt.title("Constraint enforcement with Fiacco-McCormick iterations")

    !isnothing(saveto) && plt.savefig(saveto)
    return plt.show()
end
