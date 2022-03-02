# using InfiniteOpt, ArgCheck, Ipopt
# import ControlNeuralODE as cnode

function van_der_pol_direct_phase_plot()
    u0 = [0.0, 1.0]
    tspan = (0, 1)
    Δt = 1f-1

    nodes_per_element = 2
    time_supports = 10

    # layer_sizes = (1,)
    # activations = (tanh,)
    layer_sizes = (8, 8, 1)
    activations = (tanh, tanh, identity)

    infopt_result = van_der_pol_neural_collocation(
        u0, tspan, layer_sizes, activations; nodes_per_element, time_supports
    )

    system! = VanDerPol()
    controller = (x, p) -> chain(x, p, layer_sizes, activations)
    controlODE = ControlODE(controller, system!, u0, tspan; Δt, params=infopt_result.params)

    phase_portrait(
        controlODE,
        result.params,
        square_bounds(u0, 7);
        markers=[
            InitialMarkers(; points=result.states[:, 1]),
            IntegrationPath(; points=result.states),
            FinalMarkers(; points=result.states[:, end]),
        ],
    )
    return nothing
end
