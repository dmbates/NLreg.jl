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

## Creating a PLregMod type

A `PLregMod` type a model matrix, usually called `MM`, for the
conditionally linear parameters, the three-dimensional Jacobian array,
usually called `MMD`, with each face corresponding to the partial
derivative of `MM` with respect to one of the nonlinear parameters,
and the values of any covariates needed to evaluate the model.  There
must be a method for `newpar`, which updates both arrays for a new value
of the nonlinear parameters, and a method for `pnames`.

For the Michaelis-Menten model these are
```julia
immutable MicMen{T<:Float64} <: PLregMod{T}
    x::Vector{T}
    MM::Matrix{T}
    MMD::Array{T,3}
end
MicMen{T<:Float64}(x::Vector{T}) = (n = length(x); MicMen(x, Array(T,n,1), Array(T,n,1,1)))
pnames(m::MicMen) = ["Vm", "K"]

function newpar{T<:Float64}(m::MicMen{T},K::T)
    x = m.x; MM = m.MM; MMD = m.MMD
    for i in 1:length(x)
        xi = x[i]
        denom = K + xi
        MMD[i,1,1] = -(MM[i,1] =  xi/denom)/denom
    end
    MM
end
function newpar{T<:Float64}(m::MicMen{T},nlp::Vector{T})
    length(nlp) == 1 ? newpar(m,nlp[1]) : error("length(nlp) should be 1")
end
```

A `MicMen` object contains the vector of concentrations, the model
matrix and its Jacobian.  If the vector of concentrations is of length
`n` then the model matrix is of size `n` by `1` and the Jacobian is
of size `n` by `1` by `1`.  When a new value of `K` becomes available
the model matrix is evaluated as `x./(K+x)` and the Jacobian as
`-x./abs2(K+x)`.  An explicit loop is used here to avoid allocation of
temporaries, although this is probably not necessary in any reasonable
size of data set.


