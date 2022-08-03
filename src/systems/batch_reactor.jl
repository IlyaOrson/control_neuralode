@kwdef struct BatchReactor
    α = 0.5f0
    β = 1.0f0
    γ = 1.0f0
    δ = 1.0f0
end

function (S::BatchReactor)(du, u, p, t, controller; input=:state)
    @argcheck input in (:state, :time)

    # fixed parameters
    (; α, β, γ, δ) = S

    y1, y2 = u

    # neural network outputs controls taken by the system
    if input == :state
        c1, c2 = controller(u, p)
    elseif input == :time
        c1, c2 = controller(t, p)
    end

    # dynamics of the controlled system
    y1_prime = -(c1 + α * c1^2) * y1 + δ * c2
    y2_prime = (β * c1 - γ * c2) * y1

    # update in-place
    @inbounds begin
        du[1] = y1_prime
        du[2] = y2_prime
    end
    return nothing
end

# TODO ControlODE(system::BatchReactor)
