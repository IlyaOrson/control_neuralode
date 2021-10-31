# Elements of Chemical Reaction Engineering
# Fifth Edition
# H. SCOTT FOGLER
# Chapter 13: Unsteady-State Nonisothermal Reactor Design
# Section 13.5: Nonisothermal Multiple Reactions
# Example 13–5 Multiple Reactions in a Semibatch Reactor
# p. 658

function semibatch_reactor()
    @show datadir = generate_data_subdir(@__FILE__)

    function system!(du, u, p, t, controller)
        # fixed parameters
        CpA = 30.0f0
        CpB = 60.0f0
        CpC = 20.0f0
        CpH2SO4 = 35.0f0
        N0H2S04 = 100.0f0
        T0 = 305.0f0
        CA0 = 4.0f0
        HRA = -6500.0f0
        HRB = 8000.0f0
        E1A = 9500.0f0 / 1.987f0
        E2B = 7000.0f0 / 1.987f0
        A1 = 1.25f0
        A2 = 0.08f0
        UA = 35000.0f0  # Bradford uses 45000
        # Tr1    = 420f0
        # Tr2    = 400f0

        # neural network outputs controls taken by the system
        CA, CB, CC, T, Vol = u  # state unpacking
        c_F, c_T = controller(u, p)  # control based on state and parameters

        k1A = A1 * exp(E1A * (1 / 320.0f0 - 1 / T))
        k2B = A2 * exp(E2B * (1 / 300.0f0 - 1 / T))

        k1CA = k1A * CA
        k2CB = k2B * CB
        F_Vol = c_F / Vol

        ra = -k1CA
        rb = 0.5f0 * k1CA - k2CB
        rc = 3.0f0 * k2CB

        num =
            UA * (c_T - T) - CA0 * c_F * CpA * (T - T0) +
            (HRA * (-k1CA) + HRB * (-k2CB)) * Vol
        den = (CA * CpA + CpB * CB + CpC * CC) * Vol + N0H2S04 * CpH2SO4

        # dynamics of the controlled system
        dCA = ra + (CA0 - CA) * F_Vol
        dCB = rb - CB * F_Vol
        dCC = rc - CC * F_Vol
        dT = num / den
        dVol = c_F

        # update in-place
        @inbounds begin
            du[1] = dCA
            du[2] = dCB
            du[3] = dCC
            du[4] = dT
            du[5] = dVol
        end
    end

    # initial conditions and timepoints
    t0 = 0.0f0
    tf = 0.7f0  # Bradfoard uses 0.4
    Δt = 0.02f0
    CA0 = 0.0f0
    CB0 = 0.0f0
    CC0 = 0.0f0
    T0 = 290.0f0
    V0 = 100.0f0
    u0 = [CA0, CB0, CC0, T0, V0]
    tspan = (t0, tf)
    tsteps = t0:Δt:tf

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(5, 16, tanh),
        FastDense(16, 16, tanh),
        FastDense(16, 2),
        # (x, p) -> [240f0, 298f0],
        # F ∈ (0, 250) & V ∈ (200, 500) in Bradford 2017
        (x, p) -> [250 * sigmoid(x[1]), 200 + 300 * sigmoid(x[2])],
    )

    # simulate the system with constant controls
    fogler_ref = [240.0f0, 298.0f0]  # reference values in Fogler
    fixed_dudt!(du, u, p, t) = system!(du, u, p, t, (u, p) -> fogler_ref)
    fixed_prob = ODEProblem(fixed_dudt!, u0, tspan)
    fixed_sol = solve(fixed_prob, BS3()) |> Array  # sensealg=ReverseDiffAdjoint()

    # enforce constant control over integrated path
    function precondition_loss(params)

        # this fails because Zygote does not support mutation
        # diff(state) = controller(state, params) - fogler_ref
        # coldiff = mapslices(diff, sol; dims=1)  # apply error function over columns
        # return sum(coldiff.^2)

        sum_squares = 0.0f0
        for state in eachcol(fixed_sol)
            pred = controller(state, params)
            sum_squares += sum((pred - fogler_ref) .^ 2)
        end
        return sum_squares
    end

    # destructure model weights into a vector of parameters
    θ = initial_params(controller)

    dudt!(du, u, p, t) = system!(du, u, p, t, controller)

    @info "Controls after default initialization (Xavier uniform)"
    plot_simulation(
        controller, fixed_prob, θ, tsteps; only=:controls, show=precondition_loss(θ)
    )

    adtype = GalacticOptim.AutoZygote()
    optf = GalacticOptim.OptimizationFunction((x, p) -> precondition_loss(x), adtype)
    optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
    optprob = GalacticOptim.OptimizationProblem(optfunc, θ; allow_f_increases=true)
    result = GalacticOptim.solve(optprob, ADAM(); maxiters=10)

    @info "Controls preconditioned to Fogler's reference: $(fogler_ref)"
    plot_simulation(
        controller,
        fixed_prob,
        result.minimizer,
        tsteps;
        only=:controls,
        show=precondition_loss(result.minimizer),
    )

    # constraints with barrier methods
    # T ∈ (0, 420]
    # Vol ∈ (0, 800]
    T_up = 420
    V_up = 200

    # define objective function to optimize

    function loss(params, prob, tsteps; T_up=T_up, V_up=V_up, α=1f-3, δ=1f1)

        # integrate ODE system and extract loss from result
        sol = solve(prob, BS3(); p=params, saveat=tsteps) |> Array
        out_temp = map(x -> relaxed_log_barrier(T_up - x; δ), sol[4, 1:end])
        out_vols = map(x -> relaxed_log_barrier(V_up - x; δ), sol[5, 1:end])

        last_state = sol[:, end]
        # L = - (100 x₁ - x₂) + penalty  # minus to maximize
        # return - 100f0*last_state[1] + last_state[2] + penalty
        objective = -last_state[3]

        # integral penalty
        penalty = Δt * (sum(out_temp) + sum(out_vols))

        return objective, α * penalty
    end
    prob = ODEProblem(dudt!, u0, tspan, result.minimizer)
    δ0 = 1f1
    δs = [δ0 * 0.7^i for i in 0:10]
    for δ in δs
        # global prob , result
        # local adtype , optf , optfunc , optprob

        # set differential equation struct
        # prob = ODEProblem(dudt!, u0, tspan, result.minimizer)
        prob = remake(prob; p=result.minimizer)

        # closures to comply with required interface
        loss_(params) = reduce(+, loss(params, prob, tsteps; δ))

        @info "Current Controls"
        plot_simulation(
            controller,
            prob,
            result.minimizer,
            tsteps;
            only=:controls,
            show=loss_(result.minimizer),
        )

        adtype = GalacticOptim.AutoZygote()
        optf = GalacticOptim.OptimizationFunction((x, p) -> loss_(x), adtype)
        optfunc = GalacticOptim.instantiate_function(
            optf, result.minimizer, adtype, nothing
        )
        optprob = GalacticOptim.OptimizationProblem(
            optfunc, result.minimizer; allow_f_increases=true
        )
        result = GalacticOptim.solve(
            optprob,
            ADAM();
            cb=(params, loss) -> plot_simulation(
                controller,
                prob,
                params,
                tsteps;
                only=:states,
                vars=[1, 2, 3],
                show=loss,
            ),
            maxiters=5,
        )
    end
    @info "Final states"
    plot_simulation(
        controller, prob, result.minimizer, tsteps; only=:states, vars=[1, 2, 3]
    )
    plot_simulation(
        controller,
        prob,
        result.minimizer,
        tsteps;
        only=:states,
        vars=[4, 5],
        yrefs=[T_up, V_up],
    )

    @info "Final controls"
    plot_simulation(controller, prob, result.minimizer, tsteps; only=:controls, show=loss)#  only=:states, vars=[1,2,3])

    @show final_objective, final_penalty = loss(result.minimizer, prob, tsteps; δ=δs[end])

    @info "Storing results"
    store_simulation(
        "constrained",
        datadir,
        controller,
        prob,
        result.minimizer,
        tsteps;
        metadata=Dict(
            :loss => final_objective + final_penalty,
            :objective => final_objective,
            :penalty => final_penalty,
            :num_params => length(initial_params(controller)),
            :layers => controller_shape(controller),
            :deltas => δs,
            :t0 => t0,
            :tf => tf,
            :Δt => Δt,
        ),
    )
end  # function wrapper
