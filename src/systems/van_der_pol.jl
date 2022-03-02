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
    # x3_prime = x1^2 + x2^2 + c1^2

    # update in-place
    @inbounds begin
        du[1] = x1_prime
        du[2] = x2_prime
        # du[3] = x3_prime
    end
end
