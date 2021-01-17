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

# initial conditions and timepoints
t0 = 0f0
tf = 1.2f0  # Bradfoard uses 0.4
Δt = 0.01f0
CA0 = 0f0; CB0 = 0f0; CC0 = 0f0; T0=290f0; V0 = 100f0
u0 = [CA0, CB0, CC0, T0, V0]
tspan = (t0, tf)
tsteps = t0:Δt:tf

# set arquitecture of neural network controller
controller = FastChain(
    FastDense(5, 20, tanh),
    FastDense(20, 20, tanh),
    FastDense(20, 2),
    # (x, p) -> [240f0, 298f0],
    # F ∈ (0, 250) & V ∈ (200, 500) in Bradford 2017
    (x, p) -> [250*sigmoid(x[1]), 200 + 300*sigmoid(x[2])],
)

# simulate the system with constant controls
fogler_ref = [240f0, 298f0]  # reference values in Fogler
fixed_dudt!(du, u, p, t) = system!(du, u, p, t, (u, p)->fogler_ref)
fixed_prob = ODEProblem(fixed_dudt!, u0, tspan)
fixed_sol = solve(fixed_prob, Tsit5()) |> Array

# enforce constant control over integrated path
function precondition_loss(params)
    # this fails because Zygote does not support mutation
    # diff(state) = controller(state, params) - fogler_ref
    # coldiff = mapslices(diff, sol; dims=1)  # apply error function over columns
    # return sum(coldiff.^2)

    sum_squares = 0f0
    for state in eachcol(fixed_sol)
        pred = controller(state, params)
        sum_squares += sum((pred-fogler_ref).^2)
    end
    return sum_squares
end

plot_callback(params, loss) = plot_simulation(fixed_prob, params, tsteps; only=:controls, show=loss)

# destructure model weights into a vector of parameters
θ = initial_params(controller)

@info "Controls after default initialization (Xavier uniform)"
plot_callback(θ, precondition_loss(θ))

adtype = GalacticOptim.AutoZygote()
optf = GalacticOptim.OptimizationFunction((x, p) -> precondition_loss(x), adtype)
optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, θ; allow_f_increases = true)
precondition = GalacticOptim.solve(optprob, LBFGS())

# C. Feller and C. Ebenbauer
# "Relaxed Logarithmic Barrier Function Based Model Predictive Control of Linear Systems"
# IEEE Transactions on Automatic Control, vol. 62, no. 3, pp. 1223-1238, March 2017
# doi: 10.1109/TAC.2016.2582040.

# constraints with barrier methods
# T ∈ (0, 420]
# Vol ∈ (0, 800]
β(z, δ) = 0.5f0 * (((z - 2δ)/δ)^2 - 1f0) - log(δ)
B(z; δ=0.3f0) = z > δ ? -log(z) : β(z, δ)
B(z, lower, upper; δ=10f0) = B(z - lower; δ) + B(upper - z; δ)

# define objective function to optimize
function loss(params, prob, tsteps)
    # integrate ODE system and extract loss from result
    sol = solve(prob, Tsit5(), p = params, saveat = tsteps) |> Array
    last_state = sol[:, end]
    out_temp = map(x -> B(x, 0, 400), sol[4, 1:end])
    out_vols = map(x -> B(x, 0, 380), sol[5, 1:end])
    # quadratic penalty
    penalty = sum(out_temp) + sum(out_vols)
    # L = - (100 x₁ - x₂) + penalty  # minus to maximize
    # return - 100f0*last_state[1] + last_state[2] + penalty
    return -last_state[1] + penalty
end

# set differential equation struct
dudt!(du, u, p, t) = system!(du, u, p, t, controller)
prob = ODEProblem(dudt!, u0, tspan, precondition.minimizer)

# closures to comply with required interface
loss(params) = loss(params, prob, tsteps)

@info "Controls after preconditioning to Fogler's reference: $(fogler_ref)"
plot_callback(precondition.minimizer, loss(precondition.minimizer))

adtype = GalacticOptim.AutoZygote()
optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, θ; allow_f_increases = true)
@show result = GalacticOptim.solve(
    optprob, LBFGS();
    cb = (params, loss) -> plot_simulation(prob, params, tsteps; only=:states, vars=[1,2,3], show=loss)
)

plot_simulation(prob, result.minimizer, tsteps; only=:states, vars=[4,5], show=loss)

@info "Final controls"
plot_simulation(prob, result.minimizer, tsteps; only=:states, show=loss)#  only=:states, vars=[1,2,3])

@info "Storing results"
store_simulation(@__FILE__, prob, result.minimizer, tsteps; metadata=Dict(:loss => loss(result.minimizer)))
