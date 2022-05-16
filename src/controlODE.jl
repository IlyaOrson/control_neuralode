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

    function ControlODE(
        controller,
        system,
        u0,
        tspan;
        params=initial_params(controller),
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

        # check domain types
        time_type = find_array_param(tsteps)
        space_type = find_array_param(u0)
        control_type = find_array_param(controller(u0, params))
        @argcheck space_type == control_type

        # construct ODE problem
        @assert length(methods(system)) == 1
        # number of arguments for inplace form system (du, u, p, t, controller; input)
        local prob, inplace
        if methods(system)[1].nargs < 6
            inplace = false
            dudt(u, p, t) = system(u, p, t, controller; input)
            prob = ODEProblem(dudt, u0, tspan)
        else
            inplace = true
            dudt!(du, u, p, t) = system(du, u, p, t, controller; input)
            prob = ODEProblem(dudt!, u0, tspan)
        end
        return new{space_type,time_type}(
            controller, system, u0, tspan, tsteps, integrator, sensealg, prob, inplace
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
