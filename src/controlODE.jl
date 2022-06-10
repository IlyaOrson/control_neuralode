struct ControlODE{uType<:Real,tType<:Real}
    controller
    system
    u0::AbstractVector{uType}
    tspan::Tuple{tType,tType}
    tsteps::AbstractVector{tType}
    integrator::AbstractODEAlgorithm
    sensealg::AbstractSensitivityAlgorithm
    prob::AbstractODEProblem
    inplace::Bool
    init_params::ComponentArray
    init_states::NamedTuple

    function ControlODE(
        controller,
        system,
        u0,
        tspan;
        tsteps::Union{Nothing,AbstractVector{<:Real}}=nothing,
        Δt::Union{Nothing,Real}=nothing,
        npoints::Union{Nothing,Real}=nothing,
        input::Symbol=:state,
        integrator=INTEGRATOR,
        sensealg=SENSEALG,
    )
        # check tsteps construction
        if !isnothing(tsteps)
            @argcheck tspan[begin] == tsteps[begin]
            @argcheck tspan[end] == tsteps[end]
        elseif !isnothing(Δt)
            tsteps = range(tspan...; step=Δt)
        elseif !isnothing(npoints)
            tsteps = range(tspan...; length=npoints)
        else
            # @argcheck !isnothing(tsteps) || !isnothing(Δt) || !isnothing(npoints)
            throw(
                ArgumentError(
                    "Either tsteps, Δt or npoints keyword must be provided.
                    They follow that order of priority if several are given.",
                ),
            )
        end
        # Lux is explicit with the states of a chain but we do not use them
        init_params, init_states = Lux.setup(default_rng(), controller)
        lux_controller(u, params) = controller(u, params, init_states)[1]

        # check domain types
        time_type = find_array_param(tsteps)
        space_type = find_array_param(u0)
        init_out = lux_controller(u0, init_params)
        control_type = find_array_param(init_out)
        @argcheck space_type == control_type

        # construct ODE problem
        @assert length(methods(system)) == 1
        # number of arguments for inplace form system (du, u, p, t, controller; input)
        local prob, inplace
        if methods(system)[1].nargs < 6
            inplace = false
            dudt(u, p, t) = system(u, p, t, lux_controller; input)
            prob = ODEProblem(dudt, u0, tspan)
        else
            inplace = true
            dudt!(du, u, p, t) = system(du, u, p, t, lux_controller; input)
            prob = ODEProblem(dudt!, u0, tspan)
        end
        return new{space_type,time_type}(
            lux_controller,
            system,
            u0,
            tspan,
            tsteps,
            integrator,
            sensealg,
            prob,
            inplace,
            ComponentArray(init_params),
            init_states,
        )
    end
end

# TODO follow recommended interface https://github.com/SciML/CommonSolve.jl
function solve(code::ControlODE, params; kwargs...)
    return solve(
        code.prob,
        code.integrator;
        p=params,
        saveat=code.tsteps,  # this should not be necessary
        sensealg=code.sensealg,
        kwargs...,
    )
end
