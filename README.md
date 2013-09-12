# Nonlinear regression models in [Julia](http://julialang.org)

[![Build Status](https://travis-ci.org/dmbates/NLreg.jl.png)](https://travis-ci.org/dmbates/NLreg.jl)

In this [Julia](http://julialang.org) package nonlinear regression
models are formulated as Julia types.  Partially linear models (those
models with some parameters that occur linearly in the model
expression) are expressed as types that inherit from the `PLregMod`
abstract type.  A instance of a model type is created from the values
of any covariates in the model.

## Example - the Michaelis-Menten model for enzyme kinetics

In the
[Michaelis-Menten model](http://en.wikipedia.org/wiki/Michaelis-Menten_kinetics)
for enzyme kinetics,
```julia
v = Vm * c / (K + c)
```
the relationship between the velocity, `v`, of a reaction and the
concentration, `c`, of the substrate depends on two parameters; `Vm`,
the maximum velocity and `K`, the Michaelis parameter.  The `Vm`
parameter occurs linearly in this expression whereas `K` is a
nonlinear parameter.

To fit such a model we create a `MicMen` object from the vector of
observed concentrations and a `PLregFit` object from this model and
the responses.
```julia
julia> using RDatasets, NLreg, GLM

julia> pur = data("datasets",  "Puromycin")[1:12,:]

julia> pl = PLregFit(MicMen(pur[:conc]),float(pur[:rate]));

julia> deviance(pl, [0.06]) # residual sum of squares at K = 0.06
1223.6796325318303

julia> gpinc(pl).incr          # increment
1-element Array{Float64,1}:
 0.0036186

julia> deviance(pl,pl.pars[2:2]+pl.incr)
1195.8494418228909

julia> println(pl.pars')
212.36041696796605	.06361860195046934

```

The deviance (residual sum of squares) at an initial value, `K = 0.06`
is evaluated as is the Golub-Pereyra increment, providing a new value
of `K` at which the (profiled) deviance is reduced.  This process can
be continued to convergence.

