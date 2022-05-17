RESULTS_REGEX = r"delta_(\d+\.\d+)_iter_(\d+)"

function name_interpolation(delta, iter)
    return "delta_$(round(delta, digits=2))_iter_$iter"
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

    iter_delta_dict = SortedDict{Int, Float64}()
    filepaths = readdir(dirpath; join=true, sort=true)

    for filepath in filepaths
        vals = extract_name_values(filepath)
        isnothing(vals) && continue
        iter, delta = vals
        iter_delta_dict[iter] = delta
    end
    return iter_delta_dict
end

function load_penalization_round(
    dirpath,
    system;
    iter,
    delta,
    timesteps=100
)  # system = BioReactor(), VanDerPol() ...

    filepaths = readdir(dirpath; join=true, sort=true)

    name = name_interpolation(delta, iter)
    filepath = filter_matching_filepath(filepaths, Regex(name); extension="params.jls")

    controlODE = ControlODE(system)
    weights = deserialize(filepath)
    span = controlODE.tspan[end] - controlODE.tspan[begin]
    dt = span / timesteps

    times, states, controls = run_simulation(controlODE, weights; saveat=dt)
    return times, states, controls
end
