# Elements of Chemical Reaction Engineering
# Fifth Edition
# H. SCOTT FOGLER
# Chapter 13: Unsteady-State Nonisothermal Reactor Design
# Section 13.5: Nonisothermal Multiple Reactions
# Example 13–5 Multiple Reactions in a Semibatch Reactor
# p. 658

function system!(du, u, p, t, controller)
    # fixed parameters
    CpA    = 30f0
    CpB    = 60f0
    CpC    = 20f0
    CpH2SO4= 35f0
    N0H2S04= 100f0
    T0     = 305f0
    CA0    = 4f0
    HRA    = -6500f0
    HRB    = 8000f0
    E1A    = 9500f0/1.987f0
    E2B    = 7000f0/1.987f0
    A1     = 1.25f0
    A2     = 0.08f0
    UA     = 35000f0  # Bradford uses 45000
    # where did this variables came from?
    # Tr1    = 420f0
    # Tr2    = 400f0

    # neural network outputs controls taken by the system
    CA, CB, CC, T, Vol = u  # state unpacking
    c_F, c_T = controller(u, p)  # control based on state and parameters

    k1A = A1*exp(E1A*(1/320f0 - 1/T))
    k2B = A2*exp(E2B*(1/300f0 - 1/T))

    k1CA = k1A*CA
    k2CB = k2B*CB
    F_Vol = c_F/Vol

    ra = -k1CA
    rb = 0.5f0*k1CA - k2CB
    rc = 3f0*k2CB

    num =   UA * (c_T-T) - CA0*c_F*CpA*(T-T0) +
            (HRA*(-k1CA)+HRB*(-k2CB))*Vol
    den = (CA*CpA+CpB*CB+CpC*CC)*Vol + N0H2S04*CpH2SO4

    # dynamics of the controlled system
    dCA = ra + (CA0-CA)*F_Vol
    dCB = rb - CB*F_Vol
    dCC = rc - CC*F_Vol
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

# TODO: Enfore constraints with barrier methods
# T ∈ (0, 420]
# Vol ∈ (0, 800]
outsider(x, lo, hi) = x < lo || x > hi ? x : 0

# define objective function to optimize
function loss(params, prob, tsteps)
    # integrate ODE system and extract loss from result
    sol = solve(prob, Tsit5(), p = params, saveat = tsteps) |> Array
    last_state = sol[:, end]
    temps = sol[4, 1:end]
    vols = sol[5, 1:end]
    out_temp = map(x -> outsider(x, 0, 420), temps)
    out_vols = map(x -> outsider(x, 0, 800), temps)
    # quadratic penalty
    penalty = sum(out_temp.^2) + sum(out_vols.^2)
    # L = - (100 x₁ - x₂) + penalty  # minus to maximize
    return - 100f0*last_state[1] + last_state[2] + penalty
end

# initial conditions and timepoints
t0 = 0f0
tf = 1.5f0  # Bradfoard uses 0.4
Δt = 0.04f0
CA0 = 1f0; CB0 = 0f0; CC0 = 0f0; T0=290f0; V0 = 100f0
u0 = [CA0, CB0, CC0, T0, V0]
tspan = (t0, tf)
tsteps = t0:Δt:tf

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(5, 20, relu),
    FastDense(20, 20, relu),
    FastDense(20, 2),
    (x, p) -> [240f0, 298f0],
    # (x, p) -> tanh.(x),
)

# model weights are destructured into a vector of parameters
θ = initial_params(controller)

# set differential equation struct
dudt!(du, u, p, t) = system!(du, u, p, t, controller)
prob = ODEProblem(dudt!, u0, tspan, θ)

# closures to comply with required interface
loss(params) = loss(params, prob, tsteps)
plotting_callback(params, loss) = plot_simulation(params, loss, prob, tsteps; only=:states)

# Hic sunt dracones
result = DiffEqFlux.sciml_train(
    loss,
    θ,
    # NelderMead(),
    # BFGS(initial_stepnorm = 0.01),
    LBFGS(),
    # ADAM(),
    # maxiters=20,
    cb = plotting_callback,
    allow_f_increases = true,
)
