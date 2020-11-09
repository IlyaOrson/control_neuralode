# adapted from https://diffeqflux.sciml.ai/dev/examples/feedback_control/

using DiffEqFlux, Flux, Optim, OrdinaryDiffEq
using UnicodePlots: lineplot, lineplot!

# intial conditions and timepoints
u0 = [1.0f0, 0.0f0]
tspan = (0.0f0, 1.0f0)
tsteps = 0.0f0:0.1:1.0f0

# set arquitecture of neural network controller
model_univ = FastChain(
  FastDense(2, 20, tanh),
  FastDense(20, 20, tanh),
  FastDense(20, 2),
  (x, p) -> 5*σ.(x),  # bounds for controllers
)

function dudt_univ!(du, u, p, t)
    # fixed parameters
    α, β, γ, δ = 0.5, 1.0, 1.0, 1.0

    # neural network outputs controls taken by the system
    y1, y2 = u
    c1, c2 = model_univ(u, p)

    # dynamics of the controlled system
    y1_prime = -(c1 + α * c1^2) * y1 + δ * c2
    y2_prime = (β * c1 - γ * c2) * y1

    # update in place
    du[1] = y1_prime
    du[2] = y2_prime
end

# model weights are destructured into a vector of parameters
θ = initial_params(model_univ)

# set differential equation problem and solve it
prob_univ = ODEProblem(dudt_univ!, u0, tspan, θ)
sol_univ = solve(prob_univ, Tsit5(), abstol = 1e-8, reltol = 1e-6)

# convenience function to optimize over controller parameters
function predict_univ(θ)
  return Array(solve(prob_univ, Tsit5(), p=θ, saveat = tsteps))
end

# define objective function to optimize
loss_univ(θ) = -predict_univ(θ)[2,end]  # to maximize last value of y2

# this did not work because this is the sintax for diffeq solve, not sciml
# called = false
# sv = SavedValues(Real, Tuple{Real, Real})
# function saving_call(u, t, integrator)
#   global called
#   if !called
#     @show dump(integrator)
#     called = true
#   end
#   c1, c2 = model_univ(u, integrator.p)
#   return c1, c2
# end

# handy terminal plots
function unicode_plotter(u1, u2, c1, c2)
  plt = lineplot(
    c1,
    title = "State and Control Evolution",
    name = "c1",
    xlabel = "step",
    ylim=extrema([u1; u2; c1; c2])
  )
  lineplot!(
    plt,
    c2,
    name = "c2",
    color = :blue,
  )
  lineplot!(
    plt,
    u1,
    name = "u1",
    color = :yellow,
  )
  lineplot!(
    plt,
    u2,
    name = "u2",
    color = :magenta,
  )
  return plt
end

# simulate evolution at each iteration and plot it
function plotting_callback(params, loss)
  @info "Objective" -loss
  sol = solve(prob_univ, Tsit5(), p=params, saveat = tsteps)
  u1s, u2s = Real[], Real[]
  c1s, c2s = Real[], Real[]
  for u in sol.u
    u1, u2 = u
    push!(u1s, u1)
    push!(u2s, u2)
    c1, c2 = model_univ(u, params)
    push!(c1s, c1)
    push!(c2s, c2)
  end
  display(unicode_plotter(u1s, u2s, c1s, c2s))
  sleep(5)
  return false
end

# all the magic
result_univ = DiffEqFlux.sciml_train(
  loss_univ, θ,
  BFGS(initial_stepnorm = 0.01),
  cb = plotting_callback,
  allow_f_increases = false)
