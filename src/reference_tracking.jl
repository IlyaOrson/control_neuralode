
# Hicks, G. A., & Ray, W. H. (1971).
# Approximation methods for optimal control synthesis.
# The Canadian Journal of Chemical Engineering, 49(4), 522-528.

# # irreversible reaction
# θ = 10  # final time
# cf = 1
# Tf = 300
# Tc = 290
# J = 100
# α = 1.95E-4
# k10 = 300
# N = 25.2
# u_upper = 1500
# u_lower = 0
# y1s = 0.408126
# y2s = 3.29763
# us = 370

# # reversible reaction
# θ = 1  # final time
# cf = 1
# Tf = 323
# Tc = 326
# J = 100
# α = 1
# k10 = 1.5E7
# k20 = 1.5E10
# N = 10
# γ = 1.5
# u_upper = 9.5
# u_lower = 0
# y1s = 0.433848
# y2s = 0.659684
# us = 3.234

# # case 1
# α1 = 1E6
# α2 = 2E3
# α3 = 1E-3

# # case 2
# α1 = 1E6
# α2 = 2E3
# α3 = 0

# # case 3
# α1 = 10
# α2 = 1
# α3 = 0.1

# # case 4
# α1 = 10
# α2 = 1
# α3 = 0.0

function system!(du, u, p, t, controller)

    # irreversible reaction
    time = 10f0
    cf = 1f0
    Tf = 300f0
    Tc = 290f0
    J = 100f0
    α = 1.95f-4
    k10 = 300f0
    N = 25.2f0
    yc = Tc/(J*cf)
    yf = Tf/(J*cf)

    # neural network outputs controls taken by the system
    y1, y2 = u
    c = controller(u, p)[1]  # controller output is a 1-element container

    # dynamics of the controlled system
    r = k10 * y1 * exp(-N/y2)  # - k20 * exp(-γ*N/y2) * (1-y1)
    y1_prime = (1 -y1)/time - r
    y2_prime = (yf - y2)/time + r - α*c*(y2-yc)

    # update in-place
    @inbounds begin
        du[1] = y1_prime
        du[2] = y2_prime
    end
end

# irreversible reaction
const u_upper = 1500f0
const u_lower = 0f0

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(2, 20, tanh),
    FastDense(20, 20, tanh),
    FastDense(20, 2),
    (x, p) -> [u_lower + (u_upper-u_lower) * σ(x[1])],  # controllers ∈ [u_lower, u_upper]
)

# define objective function to optimize
function loss(params, prob, tsteps)

    # case 1
    α1 = 1E6
    α2 = 2E3
    α3 = 1E-3

    # irreversible reaction
    y1s = 0.408126f0
    y2s = 3.29763f0
    us = 370f0

    sol = solve(prob, AutoTsit5(Rosenbrock23()), p=params, saveat=tsteps) |> Array # integrate ODE system

    sum_squares = 0f0
    for state in eachcol(sol)
        control = controller(state, params)
        sum_squares += α1*(control[1]-us)^2 + α2*(state[1]-y1s)^2 + α3*(state[2]-y2s)^2
    end
    return sum_squares
end

# irreversible reaction
const time = 10f0
const cf = 1f0
const Tf = 300f0
const Tc = 290f0
const J = 100f0
const yf = Tf/(J*cf)

# initial conditions and timepoints
@show u0 = [1f0, yf]
tspan = (0f0, time)
tsteps = 0f0:0.1f0:time

# model weights are destructured into a vector of parameters
θ = initial_params(controller)

# set differential equation problem and solve it
dudt!(du, u, p, t) = system!(du, u, p, t, controller)
prob = ODEProblem(dudt!, u0, tspan, θ)

plot_simulation(θ, loss, prob, tsteps, only=:controls)

# closures to comply with required interface
loss(params) = loss(params, prob, tsteps)
plotting_callback(params, loss) = plot_simulation(params, loss, prob, tsteps, only=:states)

adtype = GalacticOptim.AutoZygote()
optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, θ; allow_f_increases = true)
result = GalacticOptim.solve(optprob, LBFGS(); cb = plotting_callback)

plot_simulation(result.minimizer, loss, prob, tsteps, only=:controls)
