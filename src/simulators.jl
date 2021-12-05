function run_simulation(controller, prob, params, tsteps)

    # integrate with given parameters
    solution = OrdinaryDiffEq.solve(
        prob, AutoTsit5(Rosenbrock23()); p=params, saveat=tsteps
    )

    # construct arrays with the same type used by the integrator
    elements_type = eltype(solution.t)
    states = Array(solution)
    total_steps = size(states, 2)
    # state_dimension = size(states, 1)
    control_dimension = length(controller(solution.u[1], params))

    # regenerate controls from controller
    controls = zeros(elements_type, control_dimension, total_steps)
    for (step, state) in enumerate(solution.u)
        controls[:, step] = controller(state, params)
    end
    return solution.t, states, controls
end

function store_simulation(
    filename::Union{Nothing,String},
    controller::DiffEqFlux.FastChain,
    prob::ODEProblem,
    params::AbstractVector{<:Real},
    tsteps::AbstractVector{<:Real};
    metadata=nothing::Union{Nothing,Dict},
    datadir=nothing::Union{Nothing,String},
    store_policy=true::Bool,
)
    if isnothing(datadir) || isnothing(filename)
        @info "Results not stored due to missing filename/datadir." maxlog = 1
        return nothing
    end

    if store_policy
        bson_path = joinpath(datadir, filename * ".bson")
        BSON.@save bson_path controller

        # weights_path = joinpath(datadir, filename * "_nnweights.csv")
        # CSV.write(weights_path, Tables.table(initial_params(controller)), writeheader=false)
    end

    times, states, controls = run_simulation(controller, prob, params, tsteps)

    state_headers = ["x$i" for i in 1:size(states, 1)]
    control_headers = ["c$i" for i in 1:size(controls, 1)]

    full_data = Tables.table(
        hcat(times, states', controls'); header=vcat(["t"], state_headers, control_headers)
    )

    CSV.write(joinpath(datadir, filename * ".csv"), full_data)

    if !isnothing(metadata)
        open(joinpath(datadir, filename * "_meta.json"), "w") do f
            JSON3.pretty(f, JSON3.write(metadata))
            println(f)
        end
    end
end
