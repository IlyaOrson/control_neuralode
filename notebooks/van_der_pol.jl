### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 31b5c73e-9641-11ec-2b0b-cbd62716cc97
begin
	import Pkg
	# activate the shared project environment
	Pkg.activate(Base.current_project())
	# instantiate, i.e. make sure that all packages are downloaded
	Pkg.instantiate()
end

# ╔═╡ 07b1f884-6179-483a-8a0b-1771da59799f
begin
	using InfiniteOpt, Ipopt
	using ReverseDiff, ForwardDiff, Zygote
	using QuadGK
	using BenchmarkTools
	using UnicodePlots
end

# ╔═╡ b1756d83-125d-4eb1-af1e-9ddecb969d49
import ControlNeuralODE as cn

# ╔═╡ 028c8bc2-0106-4690-bec9-03e0ca28bdd6
cn.plt.ion()  # already done within the package
# cn.plt.style.use("fast")

# ╔═╡ 62626cd0-c55b-4c29-94c2-9d736f335349
md"# Collocation with policy"

# ╔═╡ 253458ec-12d1-4228-b57a-73c02b3b2c49
begin
	u0 = [0.0, 1.0]
    tspan = (0, 5)

    nodes_per_element = 3
    time_supports = 50

    layer_sizes = (8, 8, 1)
    activations = (tanh, tanh, x -> (tanh.(x) * 1.3 / 2) .+ 0.3)
end

# ╔═╡ 89f5b7e3-caec-4706-b818-fa49626084b4
begin
	nstates = length(u0)
    nparams = cn.count_params(nstates, layer_sizes)
    xavier_weights = cn.start_values_sampler(nstates, layer_sizes)
end

# ╔═╡ 02377f26-039d-4685-ba94-1574b3b18aa6
function vector_fun(z)
	# @show z typeof(z) z[1:nstates] z[nstates+1:end]
	x = collect(z[1:nstates])
	p = collect(z[(nstates + 1):end])
	return cn.chain(x, p, layer_sizes, activations)
end

# ╔═╡ aa575018-f57d-4180-9663-44da68d6c77c
# NOTE: JUMP does not support vector valued functions
# https://jump.dev/JuMP.jl/stable/manual/nlp/#User-defined-functions-with-vector-inputs
function scalar_fun(z...)
	return vector_fun(collect(z))
end

# ╔═╡ 639d5e58-c08a-435f-ad8e-805d10948713
cat_params = vcat(u0 , xavier_weights)

# ╔═╡ 5e130e82-8d90-4ab5-a546-5fd47f9c667c
# @benchmark ForwardDiff.gradient($(x -> vector_fun(x)[1]), $cat_params)

# ╔═╡ ceef04c1-7c0b-4e6f-ad2c-3679e6ed0055
begin
	grad_container =  similar(cat_params)
	tape =  ReverseDiff.GradientTape(x -> vector_fun(x)[1], cat_params)
	ctape = ReverseDiff.compile(tape)
	# @benchmark ReverseDiff.gradient!($grad_container, $ctape, $cat_params)
end

# ╔═╡ 52ff1192-571d-43a6-875e-7374f7316dda
# does not work
# Enzyme.gradient(Enzyme.Reverse, x -> vector_fun(x)[1], cat_params)

# ╔═╡ 09ef9cc4-2564-44fe-be1e-ce75ad189875
grad!(grad_container, params) = ReverseDiff.gradient!(grad_container, ctape, params)

# ╔═╡ d8888d92-71df-4c0e-bdc1-1249e3da23d0
function build_model(; constrain_states::Bool=true)
	optimizer = optimizer_with_attributes(
		Ipopt.Optimizer,
		"print_level" => 4,
		"tol" => 1e-2,
        "max_iter" => 10_000,
		"hessian_approximation" => "limited-memory",
		"mu_strategy" => "adaptive",
	)
	model = InfiniteModel(optimizer)
	method = OrthogonalCollocation(nodes_per_element)

	@infinite_parameter(
		model,
		t in [tspan[1], tspan[2]],
		num_supports = time_supports,
		derivative_method = method
	)
	# @infinite_parameter(model, p[1:nparams] in [-1, 1], independent = true, num_supports = param_supports)

	# TODO: remove hard constraints
	# @variable(model, -100 <= p[i=1:nparams] <= 100, start = xavier_weights[i])
	@variable(model, p[i=1:nparams], start = xavier_weights[i])

	# state variables
	@variables(
		model,
		begin
			x[1:2], Infinite(t)
		end
	)

    # initial conditions
	@constraint(model, [i = 1:2], x[i](0) == u0[i])

    if constrain_states
        @constraint(model, -0.4 <= x[1])
    end

	# https://github.com/jump-dev/MathOptInterface.jl/pull/1819
	# JuMP.register(optimizer_model(model), :scalar_fun, length(cat_params), scalar_fun; autodiff=true)  # forward mode
	JuMP.register(optimizer_model(model), :scalar_fun, length(cat_params), scalar_fun, grad!)  # reverse mode

	@constraints(
		model,
		begin
			∂(x[1], t) == (1 - x[2]^2) * x[1] - x[2] + scalar_fun(vcat(x, p)...)[1]
			∂(x[2], t) == x[1]
		end
	)

	@objective(
		model, Min, integral(x[1]^2 + x[2]^2 + scalar_fun(vcat(x, p)...)[1]^2, t)
	)
	return model
end

# ╔═╡ ebf28370-a122-46bd-84b9-e1bc6cd4ff98
@time infopt_model = build_model();

# ╔═╡ 732b8e45-fb51-454b-81d2-2d084c12df73
cn.optimize_infopt!(infopt_model; verbose=false)

# ╔═╡ a9ee1497-b0a3-447c-a989-04013d53d56c
infopt_model |> objective_value

# ╔═╡ edd395e2-58b7-41af-85ae-6af612154df5
result = cn.extract_infopt_results(infopt_model);

# ╔═╡ 1772d71a-1f7f-43cd-a4ad-0f7f54c960d0
begin
	system = cn.VanDerPol()
	controller = (x, p) -> cn.chain(x, p, layer_sizes, activations)
	controlODE = cn.ControlODE(controller, system, u0, tspan; Δt = 0.01f0)
end;

# ╔═╡ 52afbd53-5128-4482-b929-2c71398be122
function loss_discrete(controlODE, params; kwargs...)
    objective = zero(eltype(params))
    sol = cn.solve(controlODE, params; kwargs...)
	sol_arr = Array(sol)
    for i in axes(sol, 2)
        s = sol_arr[:, i]
        c = controlODE.controller(s, params)
        objective += s[1]^2 + s[2]^2 + c[1]^2
    end
    return objective * controlODE.tsteps.step
end

# ╔═╡ d964e018-1e22-44a0-baef-a18ed5979a4c
function loss_continuous(controlODE, params; kwargs...)
    objective = zero(eltype(params))
    sol = cn.solve(controlODE, params; kwargs...)
	integral, error = quadgk(
		(t)-> sum(abs2, vcat(sol(t), controlODE.controller(sol(t), params))), controlODE.tspan...
	)
    return integral
end

# ╔═╡ 42f26a4c-ac76-4212-80f9-82858ce2959c
loss_continuous(controlODE, result.params)

# ╔═╡ 5f43bf8f-5f85-4604-8cc3-2d5e8e4f9c56
cn.histogram(result.params)

# ╔═╡ a5496bf2-adb3-4036-9d02-c0fdcec95ea1
times, states, controls = cn.run_simulation(controlODE, result.params)

# ╔═╡ a0a456a0-cb5e-4866-95f6-7861d97c208c
cn.unicode_plotter(states, controls; only=:states)

# ╔═╡ 291cd17c-dad6-4d6b-a4ae-5fdfd1501833
cn.unicode_plotter(states, controls; only=:controls)

# ╔═╡ ab404ab0-1f0c-48f1-97c8-c3d1e7ec68df
function with_pyplot(f::Function)
    f()
    fig = cn.plt.gcf()
    close(fig)
	#cn.plt.clf()
    return fig
end

# ╔═╡ 92fe3d1d-9c83-4350-ae87-5e1140ec3efa
# ╠═╡ disabled = true
#=╠═╡
with_pyplot() do
	cn.phase_portrait(
		controlODE,
		result.params,
		# randn(length(result.params)),
		cn.square_bounds(u0, 7);
		markers=cn.states_markers(result.states),
	)
end
  ╠═╡ =#

# ╔═╡ 10529d5d-486e-4455-9876-5ac46768ce8a
md"# Collocation without policy"

# ╔═╡ 1a9bcfe7-23f5-4c89-946c-0a97194778f0
# ╠═╡ show_logs = false
begin
	collocation_model = cn.van_der_pol_collocation(
		controlODE.u0,
		controlODE.tspan;
		num_supports=60,
		nodes_per_element=2,
		constrain_states=true,
	)
	collocation_results = cn.extract_infopt_results(collocation_model)
	times_c, states_c, controls_c = collocation_results
	reference_controller = cn.interpolant_controller(collocation_results)
end

# ╔═╡ 2d7c7563-89c2-4807-a152-341511502e0d
cn.unicode_plotter(states_c, controls_c; only=:controls)

# ╔═╡ a567af15-e5c0-43d8-9cc6-46b53ea75d83
# ╠═╡ show_logs = false
θ_precondition = cn.preconditioner(
	controlODE,
	reference_controller;
	θ = xavier_weights,
	x_tol=1f-7,
	g_tol=1f-2,
)

# ╔═╡ 23e2fcd7-08bb-4cbf-a2ac-badd78a29e75
# ╠═╡ show_logs = false
begin
	loss_discrete(params) = loss_discrete(controlODE, params)

    @info "Training..."
    grad_zygote!(g, params) = g .= Zygote.gradient(loss_discrete, params)[1]

    θ_unconstrained = cn.optimize_ipopt(θ_precondition, loss_discrete, grad_zygote!)
end

# ╔═╡ 6704374c-70de-4e4d-9523-e516c1072348
loss_discrete(controlODE, result.params)

# ╔═╡ 9a186cb8-aca4-4124-8fed-1fcce3cc8f31
ReverseDiff.gradient(loss_discrete, xavier_weights) |> x -> sum(abs2, x)

# ╔═╡ 08def6a5-388a-481a-8073-44597d45519d
ReverseDiff.gradient(loss_discrete, θ_precondition) |> x -> sum(abs2, x)

# ╔═╡ debda675-0fc5-4a86-b9ee-9ec20f92d5f0
ReverseDiff.gradient(loss_discrete, θ_unconstrained) |> x -> sum(abs2, x)

# ╔═╡ 4e60f25b-b76d-4f16-a2aa-f8f52c425102
times_unconstrained, states_unconstrained, controls_unconstrained = cn.run_simulation(controlODE, θ_unconstrained);

# ╔═╡ 4ec2fd98-9951-475e-83ca-1e8a298c47d7
cn.unicode_plotter(states_unconstrained, controls_unconstrained; only=:controls)

# ╔═╡ 1c567a1c-7d59-4818-903d-39d1475a2bef
loss_continuous(controlODE, θ_unconstrained)

# ╔═╡ 517d33dc-f9bc-4c8d-b1b3-3d173c9eefdb
function losses(controlODE, params; α, δ, ρ)
	# integrate ODE system
	Δt = Float32(controlODE.tsteps.step)
	sol = cn.solve(controlODE, params) |> Array
	objective = 0.0f0
	control_penalty = 0.0f0
	for i in axes(sol, 2)
		s = sol[:, i]
		c = controlODE.controller(s, params)
		objective += s[1]^2 + s[2]^2
		control_penalty += c[1]^2
	end
	objective *= Δt
	control_penalty *= Δt

	# state_fault = min.(sol[1, 1:end] .+ 0.4f0, 0.0f0)
	state_fault = map(x -> cn.relaxed_log_barrier(x - -0.4f0; δ), sol[1, 1:end-1])
	# penalty = α * sum(fault .^ 2)  # quadratic penalty
	state_penalty = Δt * α * sum(state_fault)
	regularization = ρ * sum(abs2, params)
	return objective, state_penalty, control_penalty, regularization
end

# ╔═╡ cf67ae98-a86a-49f0-8a51-67fd42f7146c
# ╠═╡ show_logs = false
begin
	ρ = 0.1f0
	@time θ_constrained, barrier_progression = cn.constrained_training(
		losses,
		controlODE,
		# xavier_weights;
		θ_unconstrained;
		ρ,
		show_progressbar=false,
		datadir=nothing,
	)
end

# ╔═╡ bd88a4d0-18ed-4bfd-b0d8-8534be7b356d
losses(controlODE, xavier_weights; α=barrier_progression.α[end], δ=barrier_progression.δ[end], ρ)

# ╔═╡ 4412d9e1-478b-416f-a4a3-e277d32ed56d
losses(controlODE, θ_unconstrained; α=barrier_progression.α[end], δ=barrier_progression.δ[end], ρ)

# ╔═╡ 97c150f6-0593-4ded-afae-bb8817d133a6
losses(controlODE, θ_constrained; α=barrier_progression.α[end], δ=barrier_progression.δ[end], ρ)

# ╔═╡ 55cafd78-4dd4-4378-82d1-f716a6c4e704
lineplot(log.(barrier_progression.δ))

# ╔═╡ 9eb753d3-af33-48a9-bb45-3494c2e138bf
lineplot(log.(barrier_progression.α))

# ╔═╡ 9a866765-7a3b-402c-8563-f3f1bbbd1c57
loss_continuous(controlODE, θ_constrained)

# ╔═╡ bb9bc3cb-c6ea-41a7-87de-eb2a52fc5b5e
times_constrained, states_constrained, controls_constrained = cn.run_simulation(controlODE, θ_constrained);

# ╔═╡ ea8d4204-09e3-4355-881f-17b55f9cdcde
cn.unicode_plotter(states_constrained, controls_constrained; only=:states)

# ╔═╡ 3f393aa5-49ca-4627-a174-30dac174d79a
# ╠═╡ disabled = true
#=╠═╡
with_pyplot() do
	cn.phase_portrait(
		controlODE,
		θ_constrained,
		cn.square_bounds(controlODE.u0, 7);
		projection=[1, 2],
		markers=cn.states_markers(states_constrained),
		title="Preconditioned policy",
	)
end
  ╠═╡ =#

# ╔═╡ 2fcb5c4e-a5bf-4f26-bede-d0c53db9256d
with_pyplot() do
    function indicator(coords...)
        if coords[1] > -0.4
            return true
        end
        return false
    end
    shader = cn.ShadeConf(; indicator)
    cn.phase_portrait(
        controlODE,
        θ_constrained,
        cn.square_bounds(controlODE.u0, 7);
        shader=cn.ShadeConf(; indicator),
        projection=[1, 2],
        markers=cn.states_markers(states_constrained),
        title="Optimized policy with constraints",
    )
end

# ╔═╡ Cell order:
# ╠═31b5c73e-9641-11ec-2b0b-cbd62716cc97
# ╠═07b1f884-6179-483a-8a0b-1771da59799f
# ╠═b1756d83-125d-4eb1-af1e-9ddecb969d49
# ╠═028c8bc2-0106-4690-bec9-03e0ca28bdd6
# ╟─62626cd0-c55b-4c29-94c2-9d736f335349
# ╠═253458ec-12d1-4228-b57a-73c02b3b2c49
# ╠═89f5b7e3-caec-4706-b818-fa49626084b4
# ╠═02377f26-039d-4685-ba94-1574b3b18aa6
# ╠═aa575018-f57d-4180-9663-44da68d6c77c
# ╠═639d5e58-c08a-435f-ad8e-805d10948713
# ╠═5e130e82-8d90-4ab5-a546-5fd47f9c667c
# ╠═ceef04c1-7c0b-4e6f-ad2c-3679e6ed0055
# ╠═52ff1192-571d-43a6-875e-7374f7316dda
# ╠═09ef9cc4-2564-44fe-be1e-ce75ad189875
# ╠═d8888d92-71df-4c0e-bdc1-1249e3da23d0
# ╠═ebf28370-a122-46bd-84b9-e1bc6cd4ff98
# ╠═732b8e45-fb51-454b-81d2-2d084c12df73
# ╠═a9ee1497-b0a3-447c-a989-04013d53d56c
# ╠═edd395e2-58b7-41af-85ae-6af612154df5
# ╠═1772d71a-1f7f-43cd-a4ad-0f7f54c960d0
# ╠═52afbd53-5128-4482-b929-2c71398be122
# ╠═6704374c-70de-4e4d-9523-e516c1072348
# ╠═d964e018-1e22-44a0-baef-a18ed5979a4c
# ╠═42f26a4c-ac76-4212-80f9-82858ce2959c
# ╠═5f43bf8f-5f85-4604-8cc3-2d5e8e4f9c56
# ╠═a5496bf2-adb3-4036-9d02-c0fdcec95ea1
# ╠═a0a456a0-cb5e-4866-95f6-7861d97c208c
# ╠═291cd17c-dad6-4d6b-a4ae-5fdfd1501833
# ╟─ab404ab0-1f0c-48f1-97c8-c3d1e7ec68df
# ╠═92fe3d1d-9c83-4350-ae87-5e1140ec3efa
# ╟─10529d5d-486e-4455-9876-5ac46768ce8a
# ╠═1a9bcfe7-23f5-4c89-946c-0a97194778f0
# ╠═2d7c7563-89c2-4807-a152-341511502e0d
# ╠═a567af15-e5c0-43d8-9cc6-46b53ea75d83
# ╠═23e2fcd7-08bb-4cbf-a2ac-badd78a29e75
# ╠═9a186cb8-aca4-4124-8fed-1fcce3cc8f31
# ╠═08def6a5-388a-481a-8073-44597d45519d
# ╠═debda675-0fc5-4a86-b9ee-9ec20f92d5f0
# ╠═4e60f25b-b76d-4f16-a2aa-f8f52c425102
# ╠═4ec2fd98-9951-475e-83ca-1e8a298c47d7
# ╠═1c567a1c-7d59-4818-903d-39d1475a2bef
# ╠═517d33dc-f9bc-4c8d-b1b3-3d173c9eefdb
# ╠═cf67ae98-a86a-49f0-8a51-67fd42f7146c
# ╠═bd88a4d0-18ed-4bfd-b0d8-8534be7b356d
# ╠═4412d9e1-478b-416f-a4a3-e277d32ed56d
# ╠═97c150f6-0593-4ded-afae-bb8817d133a6
# ╠═55cafd78-4dd4-4378-82d1-f716a6c4e704
# ╠═9eb753d3-af33-48a9-bb45-3494c2e138bf
# ╠═9a866765-7a3b-402c-8563-f3f1bbbd1c57
# ╠═bb9bc3cb-c6ea-41a7-87de-eb2a52fc5b5e
# ╠═ea8d4204-09e3-4355-881f-17b55f9cdcde
# ╠═3f393aa5-49ca-4627-a174-30dac174d79a
# ╠═2fcb5c4e-a5bf-4f26-bede-d0c53db9256d
