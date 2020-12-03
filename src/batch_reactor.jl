function system!(du, u, p, t, controller)
    # fixed parameters
    α, β, γ, δ = 0.5f0, 1.0f0, 1.0f0, 1.0f0

    # neural network outputs controls taken by the system
    y1, y2 = u
    c1, c2 = controller(u, p)

    # dynamics of the controlled system
    y1_prime = -(c1 + α * c1^2) * y1 + δ * c2
    y2_prime = (β * c1 - γ * c2) * y1

    # update in-place
    @inbounds begin
        du[1] = y1_prime
        du[2] = y2_prime
    end
end

# define objective function to optimize
function loss(params, prob, tsteps)
    sol = solve(prob, Tsit5(), p = params, saveat = tsteps)  # integrate ODE system
    return -Array(sol)[2, end]  # second variable, last value, maximize
end

# initial conditions and timepoints
u0 = [1f0, 0f0]
tspan = (0f0, 1f0)
tsteps = 0f0:0.001f0:1f0

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(2, 20, tanh),
    FastDense(20, 20, tanh),
    FastDense(20, 2),
    (x, p) -> 5 * σ.(x),  # controllers ∈ (0, 5)
)

# model weights are destructured into a vector of parameters
θ = initial_params(controller)

# set differential equation problem and solve it
dudt!(du, u, p, t) = system!(du, u, p, t, controller)
prob = ODEProblem(dudt!, u0, tspan, θ)

# closures to comply with required interface
loss(params) = loss(params, prob, tsteps)
plotting_callback(params, loss) = plot_simulation(params, loss, prob, tsteps, only=:controls)

adtype = GalacticOptim.AutoZygote()
optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, θ; allow_f_increases = true)
GalacticOptim.solve(optprob, LBFGS(); cb = plotting_callback)

# https://fluxml.ai/Zygote.jl/latest/#Taking-Gradients
# an example of how to extract gradients
@show eltype(θ)
@show l₀ = loss(θ)
@time ∇θ = Zygote.gradient(loss, θ)[1]
@show typeof(∇θ)
@show eltype(∇θ)
h = 0.01f0
@show eltype(h * ∇θ)
@show lₕ = loss(θ + h * ∇θ)
@show ∇L = lₕ - l₀