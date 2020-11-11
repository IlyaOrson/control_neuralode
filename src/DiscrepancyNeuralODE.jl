# adapted from https://diffeqflux.sciml.ai/dev/examples/feedback_control/

using DiffEqFlux, Flux, Optim, OrdinaryDiffEq
using UnicodePlots: lineplot, lineplot!

function system!(du, u, p, t, controller)
    # fixed parameters
    α, β, γ, δ = 0.5, 1.0, 1.0, 1.0

    # neural network outputs controls taken by the system
    y1, y2 = u
    c1, c2 = controller(u, p)

    # dynamics of the controlled system
    y1_prime = -(c1 + α * c1^2) * y1 + δ * c2
    y2_prime = (β * c1 - γ * c2) * y1

    # update in place
    du[1] = y1_prime
    du[2] = y2_prime
end

# define objective function to optimize
function loss(params, prob, tsteps)
    sol = solve(prob, Tsit5(), p = params, saveat = tsteps)
    return -Array(sol)[2, end]  # second variable, last value, maximize
end

# handy terminal plots
function unicode_plotter(u1, u2, c1, c2)
    plt = lineplot(
        c1,
        title = "State and Control Evolution",
        name = "c1",
        xlabel = "step",
        ylim = extrema([u1; u2; c1; c2]),
    )
    lineplot!(plt, c2, name = "c2", color = :blue)
    lineplot!(plt, u1, name = "u1", color = :yellow)
    lineplot!(plt, u2, name = "u2", color = :magenta)
    return plt
end

# simulate evolution at each iteration and plot it
function plot_simulation(params, loss, prob, tsteps)
    @info "Objective" -loss
    sol = solve(prob, Tsit5(), p = params, saveat = tsteps)
    u1s, u2s = Real[], Real[]
    c1s, c2s = Real[], Real[]
    for u in sol.u
        u1, u2 = u
        push!(u1s, u1)
        push!(u2s, u2)
        c1, c2 = controller(u, params)
        push!(c1s, c1)
        push!(c2s, c2)
    end
    display(unicode_plotter(u1s, u2s, c1s, c2s))
    return false
end

# intial conditions and timepoints
u0 = [1.0f0, 0.0f0]
tspan = (0.0f0, 1.0f0)
tsteps = 0.0f0:0.1:1.0f0

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(2, 20, tanh),
    FastDense(20, 20, tanh),
    FastDense(20, 2),
    (x, p) -> 5 * σ.(x),  # bounds for controllers
)

# model weights are destructured into a vector of parameters
θ = initial_params(controller)

# set differential equation problem and solve it
dudt!(du, u, p, t) = system!(du, u, p, t, controller)
prob = ODEProblem(dudt!, u0, tspan, θ)
# sol = solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-6)

# closures to comply with required interface
loss(params) = loss(params, prob, tsteps)
plotting_callback(params, loss) = plot_simulation(params, loss, prob, tsteps)

# Hic sunt dracones
result = DiffEqFlux.sciml_train(
    loss,
    θ,
    BFGS(initial_stepnorm = 0.01),
    cb = plotting_callback,
    allow_f_increases = false,
)
