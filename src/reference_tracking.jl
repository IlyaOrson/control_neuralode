# Hicks, G. A., & Ray, W. H. (1971).
# Approximation methods for optimal control synthesis.
# The Canadian Journal of Chemical Engineering, 49(4), 522-528.

# irreversible reaction
const time = 10f0  # final time (θ in original paper)
const cf = 1f0
const Tf = 300f0
const Tc = 290f0
const J = 100f0
const α = 1.95f-4
const k10 = 300f0
const N = 25.2f0
const u_upper = 1500f0
const u_lower = 0f0
const y1s = 0.408126f0
const y2s = 3.29763f0
const us = 370f0

# # reversible reaction
# const time = 1f0  # final time (θ in original paper)
# const cf = 1f0
# const Tf = 323f0
# const Tc = 326f0
# const J = 100f0
# const α = 1f0
# const k10 = 1.5f7
# const k20 = 1.5f10
# const N = 10f0
# const γ = 1.5f0
# const u_upper = 9.5f0
# const u_lower = 0f0
# const y1s = 0.433848f0
# const y2s = 0.659684f0
# const us = 3.234f0

# adimensional constants
const yf = Tf/(J*cf)
const yc = Tc/(J*cf)

# case 1
const α1 = 1f6
const α2 = 2f3
const α3 = 1f-3

# case 2
# α1 = 1f6
# α2 = 2f3
# α3 = 0f0

# case 3
# α1 = 10f0
# α2 = 1f0
# α3 = 0.1f0

# case 4
# α1 = 10f0
# α2 = 1f0
# α3 = 0f0

function system!(du, u, p, t, controller)

    # neural network outputs controls taken by the system
    y1, y2 = u
    c = controller(u, p)[1]  # controller output is a 1-element container

    # reaction rate
    r = k10 * y1 * exp(-N/y2)  # irreversible
    # r = k10 * y1 * exp(-N/y2) - k20 * exp(-γ*N/y2) * (1-y1)  # reversible

    # dynamics of the controlled system
    y1_prime = (1 - y1)/time - r
    y2_prime = (yf - y2)/time + r - α*c*(y2-yc)

    # update in-place
    @inbounds begin
        du[1] = y1_prime
        du[2] = y2_prime
    end
end

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(2, 20, tanh),
    FastDense(20, 20, tanh),
    FastDense(20, 2),
    (x, p) -> [u_lower + (u_upper-u_lower) * σ(x[1])],  # controllers ∈ [u_lower, u_upper]
)

# define objective function to optimize
function loss(params, prob, tsteps)

    # curious error with ROS3P()
    sol = solve(prob, Tsit5(), p=params, saveat=tsteps) |> Array # integrate ODE system

    sum_squares = 0f0
    for state in eachcol(sol)
        control = controller(state, params)
        sum_squares += α1*(state[1]-y1s)^2 + α2*(state[2]-y2s)^2 + α3*(control[1]-us)^2
    end
    return sum_squares
end

# initial conditions and timepoints
@show u0 = [1f0, yf]
@show tspan = (0f0, time)
tsteps = 0f0:0.1f0:time

# model weights are destructured into a vector of parameters
θ = initial_params(controller)

# set differential equation problem and solve it
dudt!(du, u, p, t) = system!(du, u, p, t, controller)
prob = ODEProblem(dudt!, u0, tspan, θ)

# closures to comply with required interface
loss(params) = loss(params, prob, tsteps)
plotting_callback(params, loss) = plot_simulation(prob, params, tsteps; only=:states, show=loss)

plot_simulation(prob, θ, tsteps; only=:controls, show=loss(θ))

adtype = GalacticOptim.AutoZygote()
optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, θ; allow_f_increases = true)
result = GalacticOptim.solve(optprob, LBFGS(); cb = plotting_callback)

plot_simulation(prob, result.minimizer, tsteps, only=:states, vars=[1], show=loss(result.minimizer))
plot_simulation(prob, result.minimizer, tsteps, only=:states, vars=[2], show=loss(result.minimizer))
plot_simulation(prob, result.minimizer, tsteps, only=:controls, show=loss(result.minimizer))

@info "Storing results"
store_simulation(@__FILE__, prob, result.minimizer, tsteps; metadata=Dict(:loss => loss(result.minimizer)))
