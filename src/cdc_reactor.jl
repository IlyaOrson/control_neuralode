function system!(du, u, p, t, controller)
    # fixed parameters
    f = (
        CpA    = 30f0,
        CpB    = 60f0,
        CpC    = 20f0,
        CpH2SO4= 35f0,
        T0     = 305f0,
        HRA    = -6500f0,
        HRB    = 8000f0,
        E1A    = 9500f0/1.987f0,
        E2A    = 7000f0/1.987f0,
        A1     = 1.25f0,
        Tr1    = 420f0,
        Tr2    = 400f0,
        CA0    = 4f0,
        A2     = 0.08f0,
        UA     = 4.5f0,
        N0H2S04= 100f0
    )

    # neural network outputs controls taken by the system
    CA, CB, CC, T, Vol = u  # state unpacking
    c_F, c_T = controller(u, p)  # control based on state and parameters

    num = 10f0^4 * f.UA * (c_T-T) - f.CA0*c_F*f.CpA*(T-f.T0)+(f.HRA*(-f.Tr1*CA)+f.HRB*(-f.Tr2*CB))*Vol
    den = (CA*f.CpA+f.CpB*CB+f.CpC*CC)*Vol + f.N0H2S04*f.CpH2SO4

    # dynamics of the controlled system
    dCA = -f.Tr1*CA + (f.CA0-CA)*(c_F / Vol)
    dCB = f.Tr1*CA/2f0 - f.Tr2*CB - CB*(c_F/Vol)
    dCC = 3f0*f.Tr2*CB - CC*(c_F/Vol)
    dT = num/den
    dVol = c_F

    # update in-place
    @inbounds begin
        du[1] = dCA
        du[2] = dCB
        du[3] = dCC
        du[4] = dT
        du[5] = dVol
    end
end

# define objective function to optimize
function loss(params, prob, tsteps)
    # integrate ODE system and extract loss from result
    sol = solve(prob, Tsit5(), p = params, saveat = tsteps)
    last_state = Array(sol)[:, end]  # minus to maximize
    return - 100f0*last_state[1] + last_state[2]  # L = - (100 x₁ - x₂)
end

# initial conditions and timepoints
t0 = 0f0
tf = 4f0
Δt = 0.004f0
CA0 = 10f0; CB0 = 10f0; CC0 = 10f0; T0=290f0; V0 = 10f0
u0 = [CA0, CB0, CC0, T0, V0]
tspan = (t0, tf)
tsteps = t0:Δt:tf

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(5, 20, tanh),
    # FastDense(20, 20, tanh),
    # FastDense(20, 20, tanh),
    FastDense(20, 2),
    (x, p) -> relu.(x),
)

# TODO: How to enforce constraints???
# T ∈ (0, 420]
# Vol ∈ (0, 800]

# model weights are destructured into a vector of parameters
θ = initial_params(controller)

# set differential equation struct
dudt!(du, u, p, t) = system!(du, u, p, t, controller)
prob = ODEProblem(dudt!, u0, tspan, θ)

# closures to comply with required interface
loss(params) = loss(params, prob, tsteps)
plotting_callback(params, loss) = plot_simulation(params, loss, prob, tsteps, :controls)

# Hic sunt dracones
result = DiffEqFlux.sciml_train(
    loss,
    θ,
    # NelderMead(),
    # BFGS(initial_stepnorm = 0.01),
    LBFGS(),
    # maxiters=5,
    cb = plotting_callback,
    allow_f_increases = true,
)

# # https://fluxml.ai/Zygote.jl/latest/#Taking-Gradients
# # an example of how to extract gradients
# @show eltype(θ)
# @show l₀ = loss(θ)
# @time ∇θ = Zygote.gradient(loss, θ)[1]
# @show typeof(∇θ)
# @show eltype(∇θ)
# h = 1e-2  # eltype(θ)(1e-3)
# @show eltype(h * ∇θ)
# @show lₕ = loss(θ + h * ∇θ)
# @show ∇L = lₕ - l₀