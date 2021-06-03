# Bradford, E., Imsland, L., Zhang, D., & del Rio Chanona, E. A. (2020).
# Stochastic data-driven model predictive control using Gaussian processes.
# Computers & Chemical Engineering, 139, 106844.

function bioreactor()
    @show datadir = generate_data_subdir(@__FILE__)

    function system!(du, u, p, t, controller, input=:state)

        # fixed parameters
        u_m = 0.0572f0
        u_d = 0.0f0
        K_N = 393.1f0
        Y_NX = 504.5f0
        k_m = 0.00016f0
        k_d = 0.281f0
        k_s = 178.9f0
        k_i = 447.1f0
        k_sq = 23.51f0
        k_iq = 800.0f0
        K_Np = 16.89f0

        # neural network outputs controls based on state
        C_X, C_N, C_qc = u  # state unpacking
        if input == :state
            I, F_N = controller(u, p)  # control based on state and parameters
        elseif input == :time
            I, F_N = controller(t, p)  # control based on time and parameters
        else
            error("The _input_ argument should be either :state of :time")
        end

        # auxiliary variables
        I_ksi = I / (I + k_s + I^2.0f0 / k_i)
        CN_KN = C_N / (C_N + K_N)

        I_kiq = I / (I + k_sq + I^2.0f0 / k_iq)
        Cqc_KNp = C_qc / (C_N + K_Np)

        # dynamics of the controlled system
        dC_X = u_m * I_ksi * C_X * CN_KN - u_d * C_X
        dC_N = -Y_NX * u_m * I_ksi * C_X * CN_KN + F_N
        dC_qc = k_m * I_kiq * C_X - k_d * Cqc_KNp

        # update in-place
        @inbounds begin
            du[1] = dC_X
            du[2] = dC_N
            du[3] = dC_qc
        end
    end

    # initial conditions and timepoints
    t0 = 0.0f0
    tf = 240.0f0
    Δt = 10.0f0
    C_X₀, C_N₀, C_qc₀ = 1.0f0, 150.0f0, 0.0f0
    u0 = [C_X₀, C_N₀, C_qc₀]
    tspan = (t0, tf)
    tsteps = t0:Δt:tf

    control_ranges = [(120.0f0, 400.0f0), (0.0f0, 40.0f0)]
    # function scaled_sigmoids(control_ranges)
    #     control_type = control_ranges |> eltype |> eltype
    #     return (x, p) -> [mean(range) + (range[end]-range[1]) * sigmoid(x[i]) for (i, range) in enumerate(control_ranges)]
    # end

    # set arquitecture of neural network controller
    controller = FastChain(
        (x, p) -> [x[1], x[2] / 10.0f0, x[3] * 10.0f0],  # input scaling
        FastDense(3, 16, tanh; initW=(x, y) -> Float32(5 / 3) * Flux.glorot_uniform(x, y)),
        FastDense(16, 16, tanh; initW=(x, y) -> Float32(5 / 3) * Flux.glorot_uniform(x, y)),
        FastDense(16, 2; initW=(x, y) -> Float32(5 / 3) * Flux.glorot_uniform(x, y)),
        # I ∈ [120, 400] & F ∈ [0, 40] in Bradford 2020
        (x, p) -> [280.0f0 * sigmoid(x[1]) + 120.0f0, 40.0f0 * sigmoid(x[2])],
    )

    # initial parameters
    @show controller_shape(controller)
    θ = initial_params(controller)  # destructure model weights into a vector of parameters
    @time display(histogram(θ; title="Number of params: $(length(θ))"))

    # set differential equation problem
    dudt!(du, u, p, t) = system!(du, u, p, t, controller)
    prob = ODEProblem(dudt!, u0, tspan, θ)

    @info "Controls after initialization"
    @time plot_simulation(controller, prob, θ, tsteps; only=:controls)

    # preconditioning to control sequences
    function precondition(t, p)
        Zygote.ignore() do  # Zygote can't handle this alone.
            I_fun =
                (400.0f0 - 120.0f0) * sin(2.0f0π * (t - t0) / (tf - t0)) / 4 +
                (400.0f0 + 120.0f0) / 2
            F_fun = 40.0f0 * sin(π / 2 + 2.0f0π * (t - t0) / (tf - t0)) / 4 + 40.0f0 / 2
            return [I_fun, F_fun]
        end
        # return [300f0, 25f0]
    end
    # display(lineplot(x -> precondition(x, nothing)[1], t0, tf, xlim=(t0,tf)))
    # display(lineplot(x -> precondition(x, nothing)[2], t0, tf, xlim=(t0,tf)))

    @info "Controls after preconditioning"
    θ = preconditioner(
        controller,
        precondition,
        system!,
        t0,
        u0,
        tsteps[(end ÷ 10):(end ÷ 10):end];
        progressbar=false,
        control_range_scaling=[range[end] - range[1] for range in control_ranges],
    )

    # prob = ODEProblem(dudt!, u0, tspan, θ)
    prob = remake(prob; p=θ)

    plot_simulation(controller, prob, θ, tsteps; only=:controls)
    display(histogram(θ; title="Number of params: $(length(θ))"))

    store_simulation("precondition", datadir, controller, prob, θ, tsteps)

    function state_penalty_functional(
        solution_array, time_intervals; state_penalty=relaxed_log_barrier, δ=1f1
    )
        @assert size(solution_array, 2) == length(time_intervals) + 1

        ratio_X_N = 3f-2 / 800.0f0
        C_N_over = map(y -> state_penalty(800.0f0 - y; δ), solution_array[2, 1:end])
        C_X_over = map(
            (x, z) -> state_penalty(3f-2 - (1.1f-2 * x - z); δ=δ * ratio_X_N),
            solution_array[1, 1:end],
            solution_array[3, 1:end],
        )
        C_N_over_last = state_penalty(150.0f0 - solution_array[2, end]; δ)

        return sum((C_N_over .+ C_X_over)[1:(end - 1)] .* time_intervals) + C_N_over_last
    end

    # state constraints on control change
    # C_N(t) - 150 ≤ 0              t = T
    # C_N(t) − 800 ≤ 0              ∀t
    # 0.011 C_X(t) - C_qc(t) ≤ 3f-2 ∀t
    function loss(
        params,
        prob;
        state_penalty=relaxed_log_barrier,
        δ=1f1,
        α=1f-3,
        μ=(3.125f-8, 3.125f-6),
        ρ=1f-1,
        tsteps=(),
    )

        # integrate ODE system
        sol_raw = solve(prob, BS3(); p=params, saveat=tsteps, abstol=1f-1, reltol=1f-1)
        sol = Array(sol_raw)

        # approximate integral penalty
        # state_penalty = Δt * (sum(C_N_over) + sum(C_X_over)) + C_N_over_last  # for fixed timesteps
        Zygote.ignore() do
            global time_intervals = [
                sol_raw.t[i + 1] - sol_raw.t[i] for i in eachindex(sol_raw.t[1:(end - 1)])
            ]
        end

        state_penalty = α * state_penalty_functional(sol, time_intervals; δ, state_penalty)

        # penalty on change of controls
        control_penalty = 0.0f0
        for i in 1:(size(sol, 2) - 1)
            prev = controller(sol[:, i], params)
            post = controller(sol[:, i + 1], params)
            for j in 1:length(prev)
                control_penalty += μ[j] * (prev[j] - post[j])^2
            end
        end

        regularization = ρ * mean(abs2, θ)  # sum(abs2, θ)

        objective = -sol[3, end]  # maximize C_qc

        return objective, state_penalty, control_penalty, regularization
    end

    # α: penalty coefficient
    # δ: barrier relaxation coefficient
    α, δ = 1f-5, 100.0f0
    θ, δs, αs = constrained_training(controller, prob, loss, θ, α, δ; tsteps, datadir)

    final_values = NamedTuple{(
        :objective, :state_penalty, :control_penalty, :regularization
    )}(
        loss(θ, prob; δ, α, tsteps)
    )

    @info "Final states"
    # plot_simulation(controller, prob, θ, tsteps; only=:states, vars=[1], show=final_values)
    plot_simulation(
        controller,
        prob,
        θ,
        tsteps;
        only=:states,
        vars=[2],
        show=final_values,
        yrefs=[800, 150],
    )
    plot_simulation(controller, prob, θ, tsteps; only=:states, vars=[3], show=final_values)
    plot_simulation(
        controller,
        prob,
        θ,
        tsteps;
        only=:states,
        fun=(x, y, z) -> 1.1f-2x - z,
        yrefs=[3f-2],
    )

    @info "Final controls"
    plot_simulation(
        controller, prob, θ, tsteps; only=:controls, vars=[1], show=final_values
    )
    plot_simulation(
        controller, prob, θ, tsteps; only=:controls, vars=[2], show=final_values
    )

    function noisy(times, percentage; scale=1.0f0, type=:centered)
        if iszero(scale)
            scale = 0.1f0
        end
        width = percentage * scale * times
        if type == :centered
            translation = width / 2.0f0
        elseif type == :negative
            translation = width * -1.0f0
        elseif type == :positive
            translation = 0.0f0
        else
            throw(
                ArgumentError(
                    "type argument must be one of :centered, :positive or :negative"
                ),
            )
        end
        return [n * percentage * scale - translation for n in 1:times]
    end

    perturbation_specs = [
        Dict(:type => :centered, :scale => 1.0f0, :samples => 20, :percentage => 2f-2)
        Dict(:type => :centered, :scale => 150.0f0, :samples => 20, :percentage => 2f-2)
        Dict(
            :type => :positive, :scale => 0.0f0 + 5f-1, :samples => 20, :percentage => 2f-2
        )
    ]
    function initial_perturbations(prob, θ, specs)
        prob = remake(prob; p=θ)

        for (i, spec) in enumerate(specs)
            obs, spens, cpens = Float32[], Float32[], Float32[]
            perturbations = noisy(
                spec[:samples], spec[:percentage]; scale=spec[:scale], type=spec[:type]
            )

            boxplot("Δu[$i]", perturbations; title="perturbations") |> display

            for noise in perturbations
                noise_vec = zeros(typeof(noise), length(u0))
                noise_vec[i] = noise
                # @info u0 + noise_vec

                # local prob = ODEProblem(dudt!, u0 + noise_vec, tspan, θ)
                prob = remake(prob; u0=prob.u0 + noise_vec)

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
    return initial_perturbations(prob, θ, perturbation_specs)
end  # script wrapper
