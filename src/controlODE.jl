struct ControlODE{uType<:Real,tType<:Real}
    controller
    system!
    u0::AbstractVector{uType}
    tspan::Tuple{tType,tType}
    tsteps::AbstractVector{tType}
    integrator::AbstractODEAlgorithm
    sensealg::AbstractSensitivityAlgorithm
    prob::AbstractODEProblem
    function ControlODE(
        controller,
        system!,  # this must be in-place
        u0,
        tspan;
        params=initial_params(controller),
        tsteps::Union{Nothing,AbstractVector{<:Real}}=nothing,
        Δt::Union{Nothing,Real}=nothing,
        npoints::Union{Nothing,Real}=nothing,
        input::Symbol=:state,
        integrator=INTEGRATOR,  # Tsit5()
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
        dudt!(du, u, p, t) = system!(du, u, p, t, controller; input)
        prob = ODEProblem(dudt!, u0, tspan)

        return new{space_type,time_type}(
            controller, system!, u0, tspan, tsteps, integrator, sensealg, prob
        )
    end
end

# TODO follow recommended interface https://github.com/SciML/CommonSolve.jl
function solve(code::ControlODE, params; kwargs...)
    return solve(
        code.prob,
        code.integrator;
        p=params,
        saveat=code.tsteps,
        sensealg=code.sensealg,
        kwargs...,
    )
end
