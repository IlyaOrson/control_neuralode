# Elements of Chemical Reaction Engineering
# Fifth Edition
# H. SCOTT FOGLER
# Chapter 13: Unsteady-State Nonisothermal Reactor Design
# Section 13.5: Nonisothermal Multiple Reactions
# Example 13–5 Multiple Reactions in a Semibatch Reactor
# p. 658

@kwdef struct SemibatchReactor
    CpA=30.0f0
    CpB=60.0f0
    CpC=20.0f0
    CpH2SO4=35.0f0
    N0H2S04=100.0f0
    T0=305.0f0
    CA0=4.0f0
    HRA=-6500.0f0
    HRB=8000.0f0
    E1A=9500.0f0 / 1.987f0
    E2B=7000.0f0 / 1.987f0
    A1=1.25f0
    A2=0.08f0
    UA=35000.0f0  # 45000  Bradford value
    Tr1=320.0f0  # 420 Bradford value
    Tr2=290.0f0  # 400 Bradford value
end

function (S::SemibatchReactor)(du, u, p, t, controller; input=:state)
    @argcheck input in (:state, :time)

    (; CpA, CpB, CpC, CpH2SO4, N0H2S04, T0, CA0, HRA, HRB, E1A, E2B, A1, A2, UA, Tr1, Tr2) = S

    # neural network outputs controls taken by the system
    CA, CB, CC, T, Vol = u  # states
    # controls
    if input == :state
        c_F, c_T = controller(u, p)
    elseif input == :time
        c_F, c_T = controller(t, p)
    end
    k1A = A1 * exp(E1A * ((1.0f0 / Tr1) - (1.0f0 / T)))
    k2B = A2 * exp(E2B * ((1.0f0 / Tr2) - (1.0f0 / T)))

    k1CA = k1A * CA
    k2CB = k2B * CB
    F_Vol = c_F / Vol

    ra = -k1CA
    rb = 0.5f0 * k1CA - k2CB
    rc = 3.0f0 * k2CB

    num =
        UA * (c_T - T) - CA0 * c_F * CpA * (T - T0) +
        (HRA * (-k1CA) + HRB * (-k2CB)) * Vol
    den = (CA * CpA + CpB * CB + CpC * CC) * Vol + N0H2S04 * CpH2SO4

    # dynamics of the controlled system
    dCA = ra + (CA0 - CA) * F_Vol
    dCB = rb - CB * F_Vol
    dCC = rc - CC * F_Vol
    dT = num / den
    dVol = c_F

    # update in-place
    @inbounds begin
        du[1] = dCA
        du[2] = dCB
        du[3] = dCC
        du[4] = dT
        du[5] = dVol
    end
    return nothing
    # return [dCA, dCB, dCC, dT, dVol]
end

function ControlODE(system::SemibatchReactor)
    # initial conditions and timepoints
    t0 = 0.0f0
    tf = 0.4f0  # Bradfoard uses 0.4
    Δt = 0.01f0
    tspan = (t0, tf)

    # state: CA, CB, CC, T, Vol
    u0 = [1.0f0, 0.0f0, 0.0f0, 290.0f0, 100.0f0]

    # control constraints
    # F = volumetric flow rate
    # V = exchanger temperature
    # F = 240 & V = 298 in Fogler's book
    # F ∈ (0, 250) & V ∈ (200, 500) in Bradford 2017
    control_ranges = [(100.0f0, 700.0f0), (0.0f0, 400.0f0)]

    controller = FastChain(
        (x, p) -> [x[1], x[2], x[3], x[4] / 1f2, x[5] / 1f2],
        FastDense(5, 16, tanh),
        FastDense(16, 16, tanh),
        # FastDense(16, 16, tanh),
        FastDense(16, 2),
        # (x, p) -> [240f0, 298f0],
        scaled_sigmoids(control_ranges),
    )
    return ControlODE(controller, system, u0, tspan; Δt)
end
