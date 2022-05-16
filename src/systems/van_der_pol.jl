# Vassiliadis, V. S., Sargent, R. W. H., & Pantelides, C. C. (1994).
# Solution of a Class of Multistage Dynamic Optimization Problems. 2. Problems with Path Constraints.
# Industrial & Engineering Chemistry Research, 33(9), 2123–2133. https://doi.org/10.1021/ie00033a015

@kwdef struct VanDerPol
    μ=1f0
end

function (S::VanDerPol)(du, u, p, t, controller; input=:state)
    @argcheck input in (:state, :time)

    # neural network outputs the controls taken by the system
    x1, x2 = u

    if input == :state
        c1 = controller(u, p)[1]  # control based on state and parameters
    elseif input == :time
        c1 = controller(t, p)[1]  # control based on time and parameters
    end

    # dynamics of the controlled system
    x1_prime = S.μ * (1 - x2^2) * x1 - x2 + c1
    x2_prime = x1

    # update in-place
    @inbounds begin
        du[1] = x1_prime
        du[2] = x2_prime
    end
    return nothing
    # return [x1_prime, x2_prime]
end

function ControlODE(system::VanDerPol)
    # initial conditions and timepoints
    t0 = 0.0f0
    tf = 5.0f0
    u0 = [0.0f0, 1.0f0]
    tspan = (t0, tf)
    Δt = 0.1f0

    # set arquitecture of neural network controller
    controller = FastChain(
        FastDense(2, 12, tanh_fast),
        FastDense(12, 12, tanh_fast),
        FastDense(12, 1),
        (x, p) -> (1.3f0 .* sigmoid_fast.(x)) .- 0.3f0,
    )

    return ControlODE(controller, system, u0, tspan; Δt)
end
