### A Pluto.jl notebook ###
# v0.18.1

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
	using InfiniteOpt, ArgCheck, Ipopt
	import ControlNeuralODE as cn
	cn.plt.ion()
end

# ╔═╡ 253458ec-12d1-4228-b57a-73c02b3b2c49
begin
	u0 = [0.0, 1.0]
    tspan = (0, 1)

    nodes_per_element = 2
    time_supports = 10

    # layer_sizes = (1,)
    # activations = (tanh,)
    layer_sizes = (8, 8, 1)
    activations = (tanh, tanh, identity)
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

# ╔═╡ d8888d92-71df-4c0e-bdc1-1249e3da23d0
function infopt_direct()
	optimizer = optimizer_with_attributes(Ipopt.Optimizer)
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
	@variable(model, -100 <= p[i=1:nparams] <= 100, start = xavier_weights[i])

	# state variables
	@variables(
		model,
		begin
			x[1:2], Infinite(t)
		end
	)

	@constraint(model, [i = 1:2], x[i](0) == u0[i])

	scalar_fun(vcat(x, p)...)
	# @register scalar_fun(vcat(x, p)...)

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

	cn.optimize_infopt!(model)

	jump_model = optimizer_model(model)

	solution_summary(jump_model; verbose=false)
	return model
end

# ╔═╡ 96486402-9868-48d3-abcf-c5a3c242ac16
model_opt = infopt_direct()

# ╔═╡ edd395e2-58b7-41af-85ae-6af612154df5
result = cn.extract_infopt_results(model_opt);

# ╔═╡ 1772d71a-1f7f-43cd-a4ad-0f7f54c960d0
begin
	system = cn.VanDerPol()
	controller = (x, p) -> cn.chain(x, p, layer_sizes, activations)
	controlODE = cn.ControlODE(controller, system, u0, tspan; Δt = 0.1f0, params=result.params)
end

# ╔═╡ ab404ab0-1f0c-48f1-97c8-c3d1e7ec68df
function with_pyplot(f::Function)
    f()
    fig = cn.plt.gcf()
    close(fig)
    return fig
end

# ╔═╡ 92fe3d1d-9c83-4350-ae87-5e1140ec3efa
with_pyplot() do
	cn.phase_portrait(
		controlODE,
		result.params,
		cn.square_bounds(u0, 7);
		markers=[
			cn.InitialMarkers(; points=result.states[:, 1]),
			cn.IntegrationPath(; points=result.states),
			cn.FinalMarkers(; points=result.states[:, end]),
		],
	)
end

# ╔═╡ Cell order:
# ╠═31b5c73e-9641-11ec-2b0b-cbd62716cc97
# ╠═07b1f884-6179-483a-8a0b-1771da59799f
# ╠═253458ec-12d1-4228-b57a-73c02b3b2c49
# ╠═89f5b7e3-caec-4706-b818-fa49626084b4
# ╠═02377f26-039d-4685-ba94-1574b3b18aa6
# ╠═aa575018-f57d-4180-9663-44da68d6c77c
# ╠═d8888d92-71df-4c0e-bdc1-1249e3da23d0
# ╠═96486402-9868-48d3-abcf-c5a3c242ac16
# ╠═edd395e2-58b7-41af-85ae-6af612154df5
# ╠═1772d71a-1f7f-43cd-a4ad-0f7f54c960d0
# ╠═ab404ab0-1f0c-48f1-97c8-c3d1e7ec68df
# ╠═92fe3d1d-9c83-4350-ae87-5e1140ec3efa
