# Bradford, E., Imsland, L., Zhang, D., & del Rio Chanona, E. A. (2020).
# Stochastic data-driven model predictive control using Gaussian processes.
# Computers & Chemical Engineering, 139, 106844.

log_time = string_datetime()

function system!(du, u, p, t, controller, input=:state)

    # fixed parameters
    u_m = 0.0572f0
    u_d = 0f0
    K_N = 393.1f0
    Y_NX = 504.5f0
    k_m = 0.00016f0
    k_d = 0.281f0
    k_s = 178.9f0
    k_i = 447.1f0
    k_sq = 23.51f0
    k_iq = 800f0
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
    I_ksi = I/(I+k_s+I^2f0/k_i)
    CN_KN = C_N/(C_N+K_N)

    I_kiq = I/(I+k_sq+I^2f0/k_iq)
    Cqc_KNp = C_qc/(C_N+K_Np)

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
t0 = 0f0
tf = 240f0
Δt = 10f0
C_X₀, C_N₀, C_qc₀ = 1f0, 150f0, 0f0
u0 = [C_X₀, C_N₀, C_qc₀]
tspan = (t0, tf)
tsteps = t0:Δt:tf

# set arquitecture of neural network controller
controller = FastChain(
    (x, p) -> [x[1], x[2]/10, x[3]*10],  # input scaling
    FastDense(3, 16, tanh, initW = (x,y) -> (5/3)*Flux.glorot_uniform(x,y)),
    FastDense(16, 16, tanh, initW = (x,y) -> (5/3)*Flux.glorot_uniform(x,y)),
    FastDense(16, 2, initW = (x,y) -> (5/3)*Flux.glorot_uniform(x,y)),
    # I ∈ [120, 400] & F ∈ [0, 40] in Bradford 2020
    (x, p) -> [280f0*sigmoid(x[1]) + 120f0, 40f0*sigmoid(x[2])],
)

# set differential equation problem
dudt!(du, u, p, t) = system!(du, u, p, t, controller)

# destructure model weights into a vector of parameters
@show controller_shape(controller)
θ = initial_params(controller)
display(histogram(θ, title="Number of params: $(length(θ))"))

@info "Controls after default initialization (Xavier uniform)"
prob = ODEProblem(dudt!, u0, tspan, θ)
plot_simulation(prob, θ, tsteps; only=:controls)

# preconditioning to control sequences
function precondition(t, p)
    Zygote.ignore() do  # Zygote can't handle this alone.
        I_fun = (400f0-120f0) * sin(2f0π*(t-t0)/(tf-t0))/4 + (400f0+120f0)/2
        F_fun = 40f0 * sin(π/2 + 2f0π*(t-t0)/(tf-t0))/4 + 40f0/2
        return [I_fun, F_fun]
    end
    # return [300f0, 25f0]
end
# display(lineplot(x -> precondition(x, nothing)[1], t0, tf, xlim=(t0,tf)))
# display(lineplot(x -> precondition(x, nothing)[2], t0, tf, xlim=(t0,tf)))

@info "Controls after preconditioning"
θ = preconditioner(controller, precondition, system!, 5)
prob = ODEProblem(dudt!, u0, tspan, θ)
plot_simulation(prob, θ, tsteps; only=:controls)
display(histogram(θ, title="Number of params: $(length(θ))"))

store_simulation(
    @__FILE__, prob, θ, tsteps;
    current_datetime=log_time,
    filename="precondition",
)

# Feller, C., & Ebenbauer, C. (2014).
# Continuous-time linear MPC algorithms based on relaxed logarithmic barrier functions.
# IFAC Proceedings Volumes, 47(3), 2481–2488.
# https://doi.org/10.3182/20140824-6-ZA-1003.01022

β(z, δ) = exp(1f0 - z/δ) - 1f0 - log(δ)
B(z; δ=0.3f0) = max(z > δ ? -log(z) : β(z, δ), 0f0)
B(z, lower, upper; δ=(upper-lower)/2f0) = B(z - lower; δ) + B(upper - z; δ)

# state constraints on control change
# C_N(t) - 150 ≤ 0              t = T
# C_N(t) − 800 ≤ 0              ∀t
# 0.011 C_X(t) - C_qc(t) ≤ 3f-2 ∀t
function loss(params, prob; δ=1f1, α=1f0, tsteps=[])
    # integrate ODE system
    sol_raw = solve(prob, BS3(), p=params, abstol=1f-1, reltol=1f-1)  # saveat=tsteps
    sol = Array(sol_raw)
    sol_t = Array(sol_raw.t)

    ratio_X_N = 3f-2 / 800f0

    C_N_over = map(y -> B(800f0 - y; δ), sol[2, 1:end])
    C_X_over = map(
        (x, z) -> B(3f-2 - (1.1f-2*x - z);
        δ=δ*ratio_X_N), sol[1, 1:end], sol[3, 1:end]
    )
    C_N_over_last = B(150f0 - sol[2, end]; δ=δ)

    # integral penalty
    # constraint_penalty = Δt * (sum(C_N_over) + sum(C_X_over)) + C_N_over_last  # for fixed timesteps
    Zygote.ignore() do
        global delta_times = [sol_t[i+1] - sol_t[i] for i in eachindex(sol_t[1:end-1])]
    end
    # Zygote does not support this one for some reason...
    # for i in eachindex(sol_t[1:end-1])
    #     δt = sol_t[i+1] - sol_t[i]
    #     # constraint_penalty += C_N_over[i] * δt
    #     # constraint_penalty += C_X_over[i] * δt
    #     # constraint_penalty += (C_N_over[i] + C_X_over[i]) * δt
    # end
    constraint_penalty = sum((C_N_over .+ C_X_over)[1:end-1] .* delta_times) + C_N_over_last

    # penalty on change of controls
    control_penalty = 0f0
    for i in 1:size(sol, 2)-1
        prev = controller(sol[:,i], params)
        post = controller(sol[:,i+1], params)
        control_penalty += 3.125f-8 * (prev[1]-post[1])^2 + 3.125f-6 * (prev[2]-post[2])^2
    end

    regularization = 1f-1 * mean(abs2, θ)  # sum(abs2, θ)

    objective = -sol[3, end]  # maximize C_qc

    return objective, α * constraint_penalty, control_penalty, regularization
end

α, δ = 1f-5, 100f0
αs, δs = [], []
limit = 20
counter = 1
while true
    @show δ, α, tsteps
    global prob, loss, θ, limit, counter, δ, δs, α, αs, log_time, result
    local adtype, optf, optfunc, optprob

    prob = ODEProblem(dudt!, u0, tspan, θ)

    # closures to comply with optimization interface
    loss(params) = reduce(+, loss(params, prob; δ, α, tsteps))
    @info "Current states"
    plot_simulation(prob, θ, tsteps; only=:states, vars=[2], yrefs=[800, 150])
    plot_simulation(prob, θ, tsteps; only=:states, fun=(x,y,z)->1.1f-2x - z, yrefs=[3f-2])

    # function plot_callback(params, loss)
    #     plot_simulation(
    #         prob, params, tsteps; only=:states, vars=[2], show=loss, yrefs=[800, 150]
    #     )
    #     plot_simulation(prob, params, tsteps; only=:states, fun=(x,y,z)->1.1f-2x - z, yrefs=[3f-2])
    # end

    @info "Current controls"
    plot_simulation(prob, θ, tsteps; only=:controls)

    result = DiffEqFlux.sciml_train(
        loss, θ, LBFGS(linesearch=LineSearches.BackTracking());
        # cb=plot_callback,
        allow_f_increases=true,
        f_tol = 1f-3,
    )
    @show result |> typeof |> fieldnames
    θ = result.minimizer

    @show objective, state_penalty, control_penalty, regularization = loss(θ, prob; δ, α, tsteps)
    if isinf(state_penalty) || state_penalty/objective > 1f4
        δ *= 1.1
        @show α = 1f4 * abs(objective / state_penalty)
    else
        @info "Storing results"
        local  metadata = Dict(
            :objective => objective,
            :state_penalty => state_penalty,
            :control_penalty => control_penalty,
            :parameters => θ,
            :num_params => length(initial_params(controller)),
            :layers => controller_shape(controller),
            :penalty_relaxations => δs,
            :penalty_coefficients => αs,
            :t0 => t0,
            :tf => tf,
            :Δt => Δt,
        )
        store_simulation(
            @__FILE__, prob, θ, tsteps;
            current_datetime=log_time,
            filename="delta_$(round(δ, digits=2))",
            metadata=metadata
        )
        push!(αs, α)
        push!(δs, δ)
        δ *= 0.8
    end
    counter == limit ? break : counter += 1
end

final_objective, final_state_penalty, final_control_penalty, final_regularization = loss(θ, prob; δ, α, tsteps)
final_values = NamedTuple{(:objective, :state_penalty, :control_penalty, :regularization)}(loss(θ, prob; δ, α, tsteps))

@info "Final states"
# plot_simulation(prob, θ, tsteps; only=:states, vars=[1], show=final_values)
plot_simulation(prob, θ, tsteps; only=:states, vars=[2], show=final_values, yrefs=[800,150])
plot_simulation(prob, θ, tsteps; only=:states, vars=[3], show=final_values)
plot_simulation(prob, θ, tsteps; only=:states, fun=(x,y,z) -> 1.1f-2x - z, yrefs=[3f-2])

@info "Final controls"
plot_simulation(prob, θ, tsteps; only=:controls, vars=[1], show=final_values)
plot_simulation(prob, θ, tsteps; only=:controls, vars=[2], show=final_values)
