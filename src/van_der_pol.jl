# Solution of a Class of Multistage Dynamic Optimization Problems.
# 2.Problems with Path Constraints

function system!(du, u, p, t, controller)
    # neural network outputs controls taken by the system
    x1, x2, x3 = u
    c1 = controller(u, p)[1]

    # dynamics of the controlled system
    x1_prime = x2
    x2_prime = (1 - x1^2)*x2 - x1 + c1
    x3_prime = x1^2 + x2^2 + c1^2

    # update in-place
    @inbounds begin
        du[1] = x1_prime
        du[2] = x2_prime
        du[3] = x3_prime
    end
end

# define objective function to optimize
function loss(params, prob, tsteps)
    # integrate ODE system (stiff problem)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), p = params, saveat = tsteps)
    return Array(sol)[3, end]  # return last value of third variable ...to be minimized
end

# initial conditions and timepoints
t0 = 0f0; tf = 5f0
u0 = [1f0, 0f0, 0f0]
tspan = (t0, tf)
dt = 0.01f0
tsteps = t0:dt:tf

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(3, 20, tanh),
    # FastDense(20, 20, tanh),
    FastDense(20, 1),
)

# model weights are destructured into a vector of parameters
θ = initial_params(controller)

# set differential equation problem and solve it
dudt!(du, u, p, t) = system!(du, u, p, t, controller)
prob = ODEProblem(dudt!, u0, tspan, θ)

# closures to comply with required interface
loss(params) = loss(params, prob, tsteps)
plotting_callback(params, loss) = plot_simulation(
    params, loss, prob, tsteps, #only=:controls
)

adtype = GalacticOptim.AutoZygote()
optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, θ)#  ; allow_f_increases = true)
result = GalacticOptim.solve(optprob, LBFGS(); cb = plotting_callback)

@show result

# now add state constraint x2(t) > -0.4 with
function penalty_loss(params, prob, tsteps)
    # integrate ODE system (stiff problem)
    sol = Array(solve(prob, AutoTsit5(Rosenbrock23()), p = params, saveat = tsteps))
    fault = min.(sol[2, 1:end] .+ 0.4f0, 0)
    penalty = sum(fault.^2)  # quadratic penalty
    return sol[3, end] + penalty
end

adtype = GalacticOptim.AutoZygote()
optf = GalacticOptim.OptimizationFunction((x, p) -> penalty_loss(x, prob, tsteps), adtype)
optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, θ; allow_f_increases = true)
result = GalacticOptim.solve(optprob, LBFGS(); cb = plotting_callback)
