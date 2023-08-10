# Feedback Control Policies with Neural ODEs

This code showcases how a state-feedback neural policy, as commonly used in reinforcement
learning, may be used similarly in an optimal control problem while enforcing state and control constraints.

## Constrained Van der Pol problem
$$\begin{equation}
    \begin{aligned}
        \min_{\theta} \quad & J = \int_0^5 x_1^2 + x_2^2 + u^2 \,dt, \\
                           & \\
        \textrm{s.t.} \quad & \dot x_1(t) = x_1(1-x_2^2) - x_2 + u, \\
                            & \dot x_2(t) = x_1, \\
                            & u(t) = \pi_\theta^2(x_1, x_2) \\
                            & \\
                            & x(t_0) = (0, 1), \\
                            & \\
                            & x_1(t) + 0.4 \geq 0, \\
                            & -0.3 \leq u(t) \leq 1, \\
    \end{aligned}
\end{equation}$$

### Phase Space with embedded policy (before and after optimization)

![vdp_initial](https://github.com/IlyaOrson/control_neuralode/assets/12092488/1cbe6b23-71bd-4f5d-8f5a-7090ab4b4cd8)
![vdp_constrained](https://github.com/IlyaOrson/control_neuralode/assets/12092488/3d86f7a7-3b92-4495-9940-7dbc4813037d)

## JuliaCon 2021
The main ideas where presented in [this talk](https://www.youtube.com/watch?v=omS3ZngEygw) of JuliaCon2021.

[![Watch the video](https://img.youtube.com/vi/omS3ZngEygw/maxresdefault.jpg)](https://www.youtube.com/watch?v=omS3ZngEygw)

## Running the study cases
This code requires Julia 1.7.

Reproduce the environment with the required dependencies:
```julia
julia> using Pkg; Pkg.activate(;temp=true)
julia> Pkg.add(url="https://github.com/IlyaOrson/ControlNeuralODE.jl")
```

Run the test cases:

```julia
julia> using ControlNeuralODE: van_der_pol, bioreactor, batch_reactor, semibatch_reactor
julia> van_der_pol(store_results=true)
```

This will generate plots while the optimization runs and store result data in `data/`.

## Methodology (Control Vector Iteration for the Neural Policy parameters)
By substituting the control function of the problem by the output of the policy, the
weights of the controller become the new unconstrained controls of the system.
The problem becomes a parameter estimation problem where the Neural ODE adjoint method may be used
to backpropagate sensitivities with respect to functional cost.

This method was originally called the _Kelley-Bryson_ gradient procedure (developed in the 60s);
which is historically interesting due to being one of the earliest uses of backpropagation.
Its continuous time extension is known as _Control Vector Iteration_ (CVI) in the optimal control
literature, where it shares the not so great reputation of indirect methods.

Its modern implementation depends crucially on automatic differentiation to avoid the manual
derivations; one of the features that made the original versions unattractive.
This is where `DiffEqFlux.jl` and similar software shine. See the publication for a clear explanation of the technical details.

# Acknowledgements
The idea was inspired heavily by the trebuchet demo of Flux and the differentiable control
example of [DiffEqFlux](https://github.com/SciML/DiffEqFlux.jl). A similar idea was contrasted with reinforcement learning in [this work](https://github.com/samuela/ctpg). Chris Rackauckas advise was very useful.

## Citation

If you find this work helpful please consider citing the following paper:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2210.11245,
  doi = {10.48550/ARXIV.2210.11245},
  url = {https://arxiv.org/abs/2210.11245},
  author = {Sandoval, Ilya Orson and Petsagkourakis, Panagiotis and del Rio-Chanona, Ehecatl Antonio},
  keywords = {Optimization and Control (math.OC), Artificial Intelligence (cs.AI), Systems and Control (eess.SY), FOS: Mathematics, FOS: Mathematics, FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {Neural ODEs as Feedback Policies for Nonlinear Optimal Control},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
