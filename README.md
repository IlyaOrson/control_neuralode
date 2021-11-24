# Feedback Control Policies with Neural ODEs

This code showcases how a state-feedback neural policy, as commonly used in reinforcement
learning, may be used similarly in an optimal control problem while enforcing state and control constraints.

The main ideas where presented in [this talk](https://www.youtube.com/watch?v=omS3ZngEygw) of JuliaCon2021.

[![Watch the video](https://img.youtube.com/vi/omS3ZngEygw/maxresdefault.jpg)](https://www.youtube.com/watch?v=omS3ZngEygw)

## Running the test cases
This code requires Julia 1.7.

Reproduce the environment with the required dependencies:
```julia
julia> using Pkg; Pkg.activate(;temp=true)
julia> Pkg.add(url="https://github.com/IlyaOrson/ControlNeuralODE.jl")
```

Run the test cases:

```julia
julia> using ControlNeuralODE: van_der_pol, bioreactor  # batch_reactor, semibatch_reactor
julia> van_der_pol()
```

These will generate terminal plots while the optimization runs and results data in `data/`.

## Methodology
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

This is where `DiffEqFlux.jl` and similar software shine.

## Control Vector Iteration
We consider optimal control problems in the Bolza form, with a running cost and a terminal cost.
The objective is to minimize this functional cost given a ODE system with initial condition by
finding an optimal control function.

Following variational calculus, the Euler-Lagrange system of equations define the optimality
conditions as a set of boundary value problems:
* First equation is just the original ODE with initial condition.
    * This is solved forward in time since its final value is required. It may be stored, checkpointed and interpolated, or recalculated backwards. (controlled by sensitivity algorithm in DiffEqFlux.jl)
* Second equation defines the adjoint (Lagrange multiplier) evolution with a final condition.
    * This equation is defined through automatic differentiated quantities from the first ODE
* Third equation defines optimal control parameters.
    * This equation defines the gradient of the functional, and should be zero in optimality.
    * In Control Vector Iteration, it is reduced iteratively by gradient-based optimization.


The control box-constraints are hardly enforced through the last nonlinearity of the neural controller.

The state constraints are softly enforced through successive iterations of a functional relaxed barrier penalty. This avoids the formulation of constraints as sequences of arcs where different constraints are active, which require a distinct problem formulation per arc.

## Differences from other methods

### Indirect shooting
It is important to emphasize that this method does not guess the missing initial condition
of the adjoint equation, as is commonly done in indirect shooting methods.
Instead, the original system is integrated forward initially to later calculate the adjoint
equation backwards, as in CVI and the adjoint sensitivity analysis.

### Direct single shooting
Commonly the control is seed as a function of time and is split in predefined intervals, where a
polynomial parametrizes the control profile. This converts the problem into a discrete NLP problem.

# Acknowledgements
The idea was inspired heavily by the trebuchet demo of Flux and the differentiable control
example of [DiffEqFlux](https://github.com/SciML/DiffEqFlux.jl). A similar idea was contrasted with reinforcement learning in [this work](https://github.com/samuela/ctpg). Chris Rackauckas advise was very useful.

## Citation

If you find this work helpful please consider citing the following paper:
```bibtex
@article{sandoval2022NODEpolicies,
  title    = {Neural ODEs as Feedback Policies for Nonlinear Optimal Control},
  author   = {Sandoval, Ilya Orson and Petsagkourakis, Panagiotis and del Rio-Chanona, Ehecatl Antonio},
  date     = {2022},
  pubstate = {submitted},
  note     = {submitted},
}
```
