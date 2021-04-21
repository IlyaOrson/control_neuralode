# Bradford, E., Imsland, L., Zhang, D., & del Rio Chanona, E. A. (2020).
# Stochastic data-driven model predictive control using Gaussian processes.
# Computers & Chemical Engineering, 139, 106844.

log_time = string_datetime()

function system!(du, u, p, t, controller)

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
    I, F_N = controller(u, p)  # control based on state and parameters

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
Δt = 2f0
C_X₀, C_N₀, C_qc₀ = 1f0, 150f0, 0f0
u0 = [C_X₀, C_N₀, C_qc₀]
tspan = (t0, tf)
tsteps = t0:Δt:tf

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(3, 16, tanh),# initW = (x,y) -> 1f2*Flux.kaiming_normal(x,y)),
    FastDense(16, 16, tanh),# initW = (x,y) -> 1f2*Flux.kaiming_normal(x,y)),
    FastDense(16, 2),# initW = (x,y) -> 1f2*Flux.kaiming_normal(x,y)),
    # (x, p) -> [240f0, 298f0],
    # I ∈ [120, 400] & F ∈ [0, 40] in Bradford 2020
    (x, p) -> [280f0*sigmoid(x[1]) + 120f0, 40f0*sigmoid(x[2])],
)

# Feller, C., & Ebenbauer, C. (2014).
# Continuous-time linear MPC algorithms based on relaxed logarithmic barrier functions.
# IFAC Proceedings Volumes, 47(3), 2481–2488.
# https://doi.org/10.3182/20140824-6-ZA-1003.01022


β(z, δ) = exp(1f0 - z/δ) - 1f0 - log(δ)
B(z; δ=0.3f0) = max(z > δ ? -log(z) : β(z, δ), 0f0)
B(z, lower, upper; δ=(upper-lower)/2f0) = B(z - lower; δ) + B(upper - z; δ)

# state constraints and regularization on control change
# C_N(t) - 150 ≤ 0              t = T
# C_N(t) − 800 ≤ 0              ∀t
# C_qc(t) − 0.011 C_X(t) ≤ 0    ∀t
function loss(params, prob, tsteps; δ=1f1, α=1f0)
    # integrate ODE system
    sol = solve(prob, BS3(), p=params, saveat=tsteps) |> Array

    ratio_X_N = 1.1f-2 / 800f0

    C_N_over = map(x -> B(800f0 - x; δ), sol[2, 1:end])
    C_X_over = map((x, y) -> B(1.1f-2*y - x; δ=δ*ratio_X_N), sol[3, 1:end], sol[1, 1:end])
    C_N_over_last = B(150f0 - sol[2, end]; δ=δ)

    constraint_penalty = Δt * (sum(C_N_over) + sum(C_X_over)) + C_N_over_last

    # penalty on change of controls
    control_penalty = 0f0
    for i in 1:size(sol, 2)-1
        prev = controller(sol[:,i], params)
        post = controller(sol[:,i+1], params)
        control_penalty += 3.125f-8 * (prev[1]-post[1])^2 + 3.125f-6 * (prev[2]-post[2])^2
    end

    objective = -sol[3, end]  # maximize C_qc
    # @show objective, α*constraint_penalty, control_penalty
    return objective, α*constraint_penalty, control_penalty
end

# destructure model weights into a vector of parameters
θ = initial_params(controller)
display(histogram(θ, title="Number of params: $(length(θ))")); sleep(5)

# set differential equation problem
dudt!(du, u, p, t) = system!(du, u, p, t, controller)

α, δ = 1f-2, 1f2
αs, δs = [], []
limit = 12
counter = 1
while true
    @show δ, α
    global prob, loss, θ, limit, counter, δ, δs, α, log_time
    local adtype, optf, optfunc, optprob

    prob = ODEProblem(dudt!, u0, tspan, θ)

    # closures to comply with optimization interface
    loss(params) = reduce(+, loss(params, prob, tsteps; δ, α))
    function plot_callback(params, loss)
        plot_simulation(
            prob, params, tsteps; only=:states, vars=[2], show=loss, yrefs=[800, 150]
        )
        plot_simulation(prob, params, tsteps; only=:states, fun=(x,y,z)->z-1.1f-2x, yrefs=[0])
    end

    @info "Current controls"
    plot_simulation(prob, θ, tsteps; only=:controls, show=loss(θ))

    adtype = GalacticOptim.AutoZygote()
    optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
    optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
    optprob = GalacticOptim.OptimizationProblem(optfunc, θ; allow_f_increases=true)
    result = GalacticOptim.solve(optprob, LBFGS(); cb=plot_callback)
    θ = result.minimizer

    @show objective, state_penalty, control_penalty = loss(θ, prob, tsteps; δ, α)
    if isinf(state_penalty) || state_penalty/objective > 1f4
        δ *= 1.1
        @show α = 1f4 * abs(objective / state_penalty)
    else
        @info "Storing results"
        local  metadata = Dict(
            :objective => objective,
            :state_penalty => state_penalty,
            :control_penalty => control_penalty,
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
            current_datetime=log_time, filename="delta_$(round(δ, digits=2))", metadata=metadata
        )
        push!(δs, δ)
        δ *= 0.8
    end
    counter == limit ? break : counter += 1
end

final_objective, final_state_penalty, final_control_penalty = loss(θ, prob, tsteps; δ, α)
final_values = NamedTuple{(:objective, :state_penalty, :control_penalty)}(loss(θ, prob, tsteps; δ, α))

@info "Final states"
plot_simulation(prob, θ, tsteps; only=:states, vars=[1], show=final_values)
plot_simulation(prob, θ, tsteps; only=:states, vars=[2], show=final_values, yrefs=[800,150])
plot_simulation(prob, θ, tsteps; only=:states, vars=[3], show=final_values)
plot_simulation(prob, θ, tsteps; only=:states, fun=(x,y,z)->z-1.1f-2x, yrefs=[0])

@info "Final controls"
plot_simulation(prob, θ, tsteps; only=:controls, vars=[1], show=final_values)
plot_simulation(prob, θ, tsteps; only=:controls, vars=[2], show=final_values)
