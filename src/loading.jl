const RESULTS_REGEX = r"iter_(\d+)"
const WEIGHTS_EXT = ".jls"
const METADATA_EXT = ".json"

# this simple function centralizes the naming convention between saving and loading
name_interpolation(iter) = "iter_$iter"

function extract_name_value(filename; re=RESULTS_REGEX)
    matcher = match(re, filename)
    if isnothing(matcher)
        return nothing
    else
        @check length(matcher) == 1 "Pattern $re matched more than one number in $filename."
        iter_s = matcher.captures[begin]
        iter = parse(Int, iter_s)
        return iter
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

function extract_values_per_iter(dirpath)
    filepaths = readdir(dirpath; join=true, sort=true)
    if isempty(filepaths)
        error("The directory is empty.")
    end
    # iter_values_dict = SortedDict{Int, Dict}()
    iter_values_dict = SortedDict()


    for filepath in filepaths
        _, ext = splitext(filepath)
        ext !== METADATA_EXT && continue
        iter = extract_name_value(filepath)
        isnothing(iter) && continue
        open(filepath, "r") do io
            iter_values_dict[iter] = JSON3.read(io)
        end
    end
    return iter_values_dict
end

function load_penalization_round(dirpath, system; iter)  # system = BioReactor(), VanDerPol() ...
    filepaths = readdir(dirpath; join=true, sort=true)

    name = name_interpolation(iter)
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
    )
    @argcheck !all(map(x -> isnothing(x), (state_var, control_var)))  "Please specify either state_var or control_var."
    @argcheck !all(map(x -> !isnothing(x), (state_var, control_var)))  "Please specify just one of state_var or control_var."

    iter_values_dict = extract_values_per_iter(dirpath)
    if isempty(iter_values_dict)
        @error "No files matched the expected pattern: $(RESULTS_REGEX)"
        return
    end
    # pprint(iter_values_dict)
    iters = collect(keys(iter_values_dict))
    color_range = range(0.2, 1, length(iter_values_dict))
    cmap = ColorMap(palette)
    cmap_range = ColorMap(palette)(color_range)
    fig, ax = plt.subplots()

    local times
    for (iter, meta) in iter_values_dict

        controlODE, weights = load_penalization_round(dirpath, system; iter)
        span = controlODE.tspan[end] - controlODE.tspan[begin]
        times, states, controls = run_simulation(controlODE, weights; saveat=span / timesteps)

        α = meta[:α]
        δ = meta[:δ]

        if !isnothing(state_var)
            ax.plot(
                times,
                states[state_var,:];
                color=cmap_range[iter, :],
                label="α=$α δ=$δ",
                alpha=transparency,
            )
            plt.ylabel("x_$state_var")
        elseif !isnothing(control_var)
            ax.plot(
                times,
                controls[control_var,:];
                color=cmap_range[iter, :],
                label="α=$α δ=$δ",
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

    norm = matplotlib.colors.Normalize(; vmin=iters[begin], vmax=iters[end])
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
    return fig
end
