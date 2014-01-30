# Nonlinear regression models in [Julia](http://julialang.org)

[![Build Status](https://travis-ci.org/dmbates/NLreg.jl.png)](https://travis-ci.org/dmbates/NLreg.jl)

In this [Julia](http://julialang.org) package nonlinear regression models are formulated as Julia types that inherit from `NLregMod`.
A simple example is the predicted concentration in a 1 compartment model with a single bolus dose at time 0.
```jl
conc = exp(logV) * exp(-exp(logK)*t)
```
where `logV` and `logK` are the logarithms of the volume of distribution and `K` is the elimination rate constant and `t` is the time of the measurement.

The `logsd1` type represents this model and the data to which it is to be fit.
The fields of this type are `t`, the vector of times at which samples are drawn, `y`, the vector of measured concentrations, `mu` the mean responses at the current parameter values, `resid` the residuals at the current parameter values, and `tgrad`, the transpose of the gradient matrix.
The external constructors for this model allow it to be specified from `t` and `y` or in a Formula/Data specification.

A nonlinear regression model must provide methods for `pnames`, the parameter names, `updtmu`, update the mean response, residuals and `tgrad` from new parameter values, and `initpars`, determine initial parameter estimates from the data.

```jl
julia> using DataFrames, NLreg, StatsBase

julia> const sd1 = within(readtable(Pkg.dir("NLreg","data","sd1.csv.gz")),:(ID = pool(ID)));

julia> nl = NonlinearLS(logsd1(:(CONC ~ TIME), sd1))
Model fit by nonlinear least squares to 580 observations

2x4 DataFrame:
        parameter estimate    stderr  t_value
[1,]       "logV" 0.133559 0.0433655  3.07985
[2,]       "logK" -1.40385 0.0823891 -17.0392

Residual sum of squares at estimates = 110.59728690764713
Residual standard error = 0.43742975097431563 on 578 degrees of freedom
```

## Plans for the near future

- Nonlinear mixed-effects models fit using the Laplace approximation to the log-likelihood

- Specification of partially linear models

- Composite models consisting of a parameter transformation and a nonlinear model.

## Partially linear models (this used to work but is now broken)

Partially linear models (those models with some parameters that occur
linearly in the model expression) are expressed as types that inherit
from the `PLregMod` abstract type.  A instance of a model type is
created from the values of any covariates in the model.

## Example - a Michaelis-Menten fit

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
julia> using RDatasets, NLreg

julia> purtrt = sub(dataset("datasets","Puromycin"),:(State .== "treated"));

julia> pm1 = fit(MicMen(Conc ~ Rate, purtrt),true)
Iteration:  1, rss = 0.188744, cvg = 0.0888416 at [-0.0786133,-220.728]
   Incr: [-4.66305]  f = 1.0, rss = 0.173277
Iteration:  2, rss = 0.173277, cvg = 0.00102418 at [-0.0995101,-225.391]
   Incr: [-0.688546]  f = 1.0, rss = 0.173117
Iteration:  3, rss = 0.173117, cvg = 6.54049e-6 at [-0.10249,-226.08]
   Incr: [0.0574836]  f = 1.0, rss = 0.173116
Iteration:  4, rss = 0.173116, cvg = 7.53229e-8 at [-0.102242,-226.022]
   Incr: [-0.00614653]  f = 1.0, rss = 0.173116
Iteration:  5, rss = 0.173116, cvg = 8.30647e-10 at [-0.102269,-226.028]
   Incr: [0.000645718]  f = 1.0, rss = 0.173116

Nonlinear least squares fit to 12 observations

      Estimate Std.Error  t value Pr(>|t|)
Vm   -0.102266 0.0315309 -3.24335   0.0088
K     -226.028   7.08463 -31.9039  2.2e-11

Residual sum of squares at estimates: 0.173116
Residual standard error = 0.131574 on 10 degrees of freedom
```

## Creating a PLregMod type

A `PLregMod` type contains the transposed gradient, usually called
`tgrad` with the conditionally linear parameters first, the
three-dimensional Jacobian array, usually called `MMD`, with each face
corresponding to the partial derivative of the conditionally linear
rows of `tgrad` with respect to the nonlinear parameters, and the
values of any covariates needed to evaluate the model.  The
model-matrix function, `mmf`, is a function of two read-only
arguments; `nlp`, the nonlinear parameters in the model function as a
vector and x the covariates for a single observation, also as a
vector, and two arguments, `tg` and `MMD` that are updated in the
function.  The `tg` vector is updated with the model matrix for the
conditionally linear parameters and the `MMD` slice, considered as a
matrix, is updated with the gradient.

For the Michaelis-Menten model the model-matrix function is
```julia
function MicMenmmf(nlp,x,tg,MMD)
    x1 = x[1]
    denom = nlp[1] + x1
    MMD[1,1] = -(tg[1] =  x1/denom)/denom
end
```
The arguments are untyped, to allow for submatrices or views of
matrices and arrays, but they should be treated as three vectors and a
matrix, for the purposes of indexing.



