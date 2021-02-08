# Solution of a Class of Multistage Dynamic Optimization Problems.
# 2.Problems with Path Constraints

function system!(du, u, p, t, controller)
    # neural network outputs controls taken by the system
    x1, x2, x3 = u
    c1 = controller(u, p)[1]

    # dynamics of the controlled system
    x1_prime = (1 - x2^2)*x1 - x2 + c1
    x2_prime = x1
    x3_prime = x1^2 + x2^2 + c1^2

    # update in-place
    @inbounds begin
        du[1] = x1_prime
        du[2] = x2_prime
        du[3] = x3_prime
    end
end

# initial conditions and timepoints
t0 = 0f0; tf = 5f0
u0 = [0f0, 1f0, 0f0]
tspan = (t0, tf)
const dt = 0.1f0
tsteps = t0:dt:tf

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(3, 16, tanh),
    FastDense(16, 16, tanh),
    FastDense(16, 1),
    (x, p) -> (1.3f0 .* σ.(x)) .- 0.3f0,
)

# model weights are destructured into a vector of parameters
θ = initial_params(controller)

# set differential equation problem and solve it
dudt!(du, u, p, t) = system!(du, u, p, t, controller)

# closures to comply with required interface
prob = ODEProblem(dudt!, u0, tspan, θ)
plotting_callback(params, loss) = plot_simulation(prob, params, tsteps; only=:states, vars=[1], show=loss)

### define objective function to optimize
function loss(params, prob, tsteps)
    # integrate ODE system (stiff problem)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), p = params, saveat = tsteps)
    return Array(sol)[3, end]  # return last value of third variable ...to be minimized
end
loss(params) = loss(params, prob, tsteps)

adtype = GalacticOptim.AutoZygote()
optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, θ ; allow_f_increases = true)
result = GalacticOptim.solve(optprob, LBFGS(); cb = plotting_callback)

@show result
store_simulation(@__FILE__, prob, result.minimizer, tsteps; metadata=Dict(:loss => loss(result.minimizer), :constraint => "none"))

### now add state constraint x2(t) > -0.4 with
function penalty_loss(params, prob, tsteps, α)
    # integrate ODE system (stiff problem)
    sol = Array(solve(prob, AutoTsit5(Rosenbrock23()), p = params, saveat = tsteps))
    fault = min.(sol[1, 1:end] .+ 0.4f0, 0)
    # α = 10f0
    penalty = α * dt * sum(fault.^2)  # quadratic penalty
    return sol[3, end] + penalty
end

penalty_coefficients = [10f0, 10^2f0, 10^3f0, 10^4f0]
for α in penalty_coefficients
    global result, penalty_loss
    @show result
    # @show α

    # set differential equation struct again
    constrained_prob = ODEProblem(dudt!, u0, tspan, result.minimizer)
    plotting_callback(params, loss) = plot_simulation(constrained_prob, params, tsteps; only=:states, vars=[1], show=loss)

    # closures to comply with interface
    penalty_loss(params) = penalty_loss(params, constrained_prob, tsteps, α)

    @info "Initial Control"
    plot_simulation(constrained_prob, result.minimizer, tsteps; only=:controls, show=penalty_loss(result.minimizer))

    adtype = GalacticOptim.AutoZygote()
    optf = GalacticOptim.OptimizationFunction((x, p) -> penalty_loss(x, constrained_prob, tsteps, α), adtype)
    optfunc = GalacticOptim.instantiate_function(optf, result.minimizer, adtype, nothing)
    optprob = GalacticOptim.OptimizationProblem(optfunc, result.minimizer; allow_f_increases = true)
    result = GalacticOptim.solve(optprob, LBFGS(); cb = plotting_callback)
end

@info "Storing results"
constrained_prob = ODEProblem(dudt!, u0, tspan, result.minimizer)
store_simulation(@__FILE__, constrained_prob, result.minimizer, tsteps; metadata=Dict(:loss => penalty_loss(result.minimizer, constrained_prob, tsteps, penalty_coefficients[end]), :constraint => "quadratic x2(t) > -0.4"))
