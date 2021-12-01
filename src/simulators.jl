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

function initial_conditions_variations(
    loss::Function, controller::Function, prob, params, tsteps, datadir, u0, δu, N, M
)
    @showprogress for n in 1:N, m in 1:M
        initial_condition = u0 + [n * δu[1], m * δu[2]]
        # prob = ODEProblem(dudt!, u0, tspan, θ)
        prob = remake(prob; u0=initial_condition)
        objective = loss(params, prob, tsteps)
        store_simulation(
            "u0+($n,$m)δu",
            controller,
            prob,
            params,
            tsteps;
            datadir,
            store_policy=false,
            metadata=Dict(
                :loss => objective,
                :u0 => initial_condition,
                :u0_original => u0,
                :δu => δu,
                :N => N,
                :M => M,
            ),
        )
    end
end

function initial_perturbations(controller, prob, θ, tsteps, u0, specs)
    prob = remake(prob; p=θ)

    fig, axs = plt.subplots(length(specs), 2; sharex="col", squeeze=false, constrained_layout=true)
    for (i, spec) in enumerate(specs)
        obs, spens, cpens = Float32[], Float32[], Float32[]
        perturbations = local_grid(
            spec[:samples], spec[:percentage]; scale=spec[:scale], type=spec[:type]
        )

        boxplot("Δu[$i]", perturbations; title="perturbations") |> display

        for noise in perturbations
            noise_vec = zeros(typeof(noise), length(u0))
            noise_vec[i] = noise
            # @info u0 + noise_vec

            # local prob = ODEProblem(dudt!, u0 + noise_vec, tspan, θ)
            prob = remake(prob; u0=prob.u0 + noise_vec)

            times, states, controls = run_simulation(controller, prob, θ, tsteps)

            cmap = ColorMap("tab10")
            axs[1,1].set_title("States")
            axs[1,2].set_title("Controls")
            axs[end,1].set_xlabel("time")
            axs[end,2].set_xlabel("time")
            for s in 1:size(states,1)
                axs[s,1].plot(times, states, label="s$s", c=cmap(s))
                axs[s,1].set_ylabel("u0 + $noise_vec")
            end
            for c in 1:size(controls, 1)
                axs[c,2].plot(times, states, label="c$c", c=cmap(c + size(states, 1)))
            end
            fig.suptitle("Initial condition noise")
            fig.show()

            objective, state_penalty, control_penalty, _ = loss(
                θ, prob; tsteps, state_penalty=indicator_function
            )
            # plot_simulation(controller, prob, θ, tsteps; only=:states, vars=[2], yrefs=[800,150])
            # plot_simulation(controller, prob, θ, tsteps; only=:states, fun=(x,y,z) -> 1.1f-2x - z, yrefs=[3f-2])

            push!(obs, objective)
            push!(spens, state_penalty)
            push!(cpens, control_penalty)
        end
        try  # this fails when penalties explode due to the steep barriers
            boxplot(
                ["objectives", "state_penalties", "constraint_penalties"],
                [obs, spens, cpens];
                title="Perturbation results",
            ) |> display
            lineplot(
                prob.u0[i] .+ perturbations, obs; title="u0 + Δu[$i] ~ objectives"
            ) |> display
            lineplot(
                prob.u0[i] .+ perturbations,
                spens;
                title="u0 + Δu[$i] ~ state_penalties",
            ) |> display
            lineplot(
                prob.u0[i] .+ perturbations,
                cpens;
                title="u0 + Δu[$i] ~ constraint_penalties",
            ) |> display
        catch
            @show obs
            @show spens
            @show cpens
        end
    end
end
