# Bradford, E., Imsland, L., Zhang, D., & del Rio Chanona, E. A. (2020).
# Stochastic data-driven model predictive control using Gaussian processes.
# Computers & Chemical Engineering, 139, 106844.

@kwdef struct BioReactor
    u_m=0.0572f0
    u_d=0.0f0
    K_N=393.1f0
    Y_NX=504.5f0
    k_m=0.00016f0
    k_d=0.281f0
    k_s=178.9f0
    k_i=447.1f0
    k_sq=23.51f0
    k_iq=800.0f0
    K_Np=16.89f0
end

function (S::BioReactor)(du, u, p, t, controller; input=:state)
    @argcheck input in (:state, :time)

    (; u_m, u_d, K_N, Y_NX, k_m, k_d, k_s, k_i, k_sq, k_iq, K_Np) = S

    C_X, C_N, C_qc = u

    if input == :state
        I, F_N = controller(u, p)
    elseif input == :time
        I, F_N = controller(t, p)
    end

    # auxiliary variables
    I_ksi = I / (I + k_s + I^2.0f0 / k_i)
    CN_KN = C_N / (C_N + K_N)

    I_kiq = I / (I + k_sq + I^2.0f0 / k_iq)
    Cqc_KNp = C_qc / (C_N + K_Np)

    # dynamics of the controlled system
    dC_X = u_m * I_ksi * C_X * CN_KN - u_d * C_X
    dC_N = -Y_NX * u_m * I_ksi * C_X * CN_KN + F_N
    dC_qc = k_m * I_kiq * C_X - k_d * Cqc_KNp

    # update in-place
    @inbounds begin
        du[1] = dC_X
        du[2] = dC_N
        du[3] = dC_qc
    end
    return nothing
    # return [dC_X, dC_N, dC_qc]
end

function ControlODE(system::BioReactor)
    # initial conditions and timepoints
    t0 = 0.0
    tf = 240.0
    Δt = 2.0
    tspan = (t0, tf)
    u0 = [1.0, 150.0, 0.0]  # C_X₀, C_N₀, C_qc₀
    control_constraints = [(80.0, 180.0), (0.0, 20.0)]

    # weights initializer reference https://pytorch.org/docs/stable/nn.init.html
    controller = FastChain(
        (x, p) -> [x[1], x[2] / 100.0, x[3] * 10.0],  # input scaling
        FastDense(3, 32, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        FastDense(32, 32, tanh_fast; initW=(x, y) -> Float32(5 / 3) * glorot_uniform(x, y)),
        FastDense(32, 2; initW=(x, y) -> glorot_uniform(x, y)),
        # I ∈ [120, 400] & F ∈ [0, 40] in Bradford 2020
        # (x, p) -> [280f0 * sigmoid(x[1]) + 120f0, 40f0 * sigmoid(x[2])],
        scaled_sigmoids(control_constraints),
    )
    return ControlODE(controller, system, u0, tspan; Δt)
end
