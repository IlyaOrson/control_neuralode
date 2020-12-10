# Bradford, E., Imsland, L., Zhang, D., & del Rio Chanona, E. A. (2020).
# Stochastic data-driven model predictive control using Gaussian processes.
# Computers & Chemical Engineering, 139, 106844.

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

outsider(x, lo, hi) = x < lo || x > hi ? x : zero(x)

# state constraints and regularization on control change
# C_N(t) - 150 ≤ 0              t = T
# C_N(t) − 800 ≤ 0              ∀t
# C_qc(t) − 0.011 C_X(t) ≤ 0    ∀t
function penalty_loss(params, prob, tsteps; β=0.1f0)
    # integrate ODE system (stiff problem)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), p = params, saveat = tsteps) |> Array
    C_N_over = relu.(sol[2, 1:end] .- 800f0)
    C_X_over = relu.(sol[3, 1:end] .- 0.011f0*sol[1, 1:end])
    C_N_over_last = relu(sol[2, end] - 150f0)

    # TODO: add penalty on change of controls

    constraint_penalty = sum(C_N_over.^2 .+ C_X_over.^2 .+ C_N_over_last.^2)
    return -sol[3, end] + β*constraint_penalty
end

# initial conditions and timepoints
t0 = 0f0
tf = 240f0  # Bradfoard uses 0.4
Δt = 1f0
C_X₀, C_N₀, C_qc₀ = 1f0, 150f0, 0f0
u0 = [C_X₀, C_N₀, C_qc₀]
tspan = (t0, tf)
tsteps = t0:Δt:tf

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(3, 20, tanh),
    FastDense(20, 20, tanh),
    FastDense(20, 2),
    # (x, p) -> [240f0, 298f0],
    # I ∈ [120, 400] & F ∈ [0, 40] in Bradford 2020
    (x, p) -> [180f0*sigmoid(x[1]) + 120f0, 40*sigmoid(x[2])],
)

# destructure model weights into a vector of parameters
θ = initial_params(controller)

# set differential equation problem
dudt!(du, u, p, t) = system!(du, u, p, t, controller)
prob = ODEProblem(dudt!, u0, tspan, θ)

# closures to comply with optimization interface
penalty_loss(params) = penalty_loss(params, prob, tsteps, )
plot_states_callback(params, loss) = plot_simulation(params, loss, prob, tsteps; only=:states, vars=[1,3])


@info "Initial controls"
plot_simulation(θ, penalty_loss(θ), prob, tsteps; only=:controls)


adtype = GalacticOptim.AutoZygote()
optf = GalacticOptim.OptimizationFunction((x, p) -> penalty_loss(x, prob, tsteps), adtype)
optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, θ; allow_f_increases = true)
result = GalacticOptim.solve(optprob, LBFGS(); cb = plot_states_callback)

@info "Final states"
plot_simulation(result.minimizer, penalty_loss(result.minimizer), prob, tsteps; only=:states)

@info "Final controls"
plot_simulation(result.minimizer, penalty_loss(result.minimizer), prob, tsteps; only=:controls)
