# Nonlinear regression

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dmbates.github.io/NLreg.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dmbates.github.io/NLreg.jl/dev)
[![Build Status](https://travis-ci.com/dmbates/NLreg.jl.svg?branch=master)](https://travis-ci.com/dmbates/NLreg.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/dmbates/NLreg.jl?svg=true)](https://ci.appveyor.com/project/dmbates/NLreg-jl)
[![Codecov](https://codecov.io/gh/dmbates/NLreg.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dmbates/NLreg.jl)

This package is an experiment in using the [`Zygote`](https://github.com/FluxML/Zygote.jl) automatic differentiation package and the `lowrankupdate!` function in the `LinearAlgebra` package to solve the linear least squares problem for a Gauss-Newton update.

The data are represented as a `Tables.RowTable`, which is a vector of `NamedTuple`s.  The model parameters are also a `NamedTuple`.  The model function is given as a function of two arguments - the parameters and a data row.

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
julia> using CSV, DataFrames, NLreg

julia> datadir = normpath(joinpath(dirname(pathof(NLreg)), "..", "data"));

julia> PurTrt = first(groupby(CSV.read(joinpath(datadir, "Puromycin.csv")), :state))
12×3 SubDataFrame
│ Row │ conc    │ rate    │ state   │
│     │ Float64 │ Float64 │ String  │
├─────┼─────────┼─────────┼─────────┤
│ 1   │ 0.02    │ 76.0    │ treated │
│ 2   │ 0.02    │ 47.0    │ treated │
│ 3   │ 0.06    │ 97.0    │ treated │
⋮
│ 9   │ 0.56    │ 191.0   │ treated │
│ 10  │ 0.56    │ 201.0   │ treated │
│ 11  │ 1.1     │ 207.0   │ treated │
│ 12  │ 1.1     │ 200.0   │ treated │

julia> pm1 = fit(NLregModel, PurTrt, :rate, (p,d) -> p.Vm * d.conc/(p.K + d.conc),
                  (Vm = 200., K = 0.05))
Nonlinear regression model fit by maximum likelihood

Data schema (response variable is rate)
Tables.Schema:
 :conc   Float64
 :rate   Float64
 :state  String

 Sum of squared residuals at convergence: 1195.4488145417758
 Achieved convergence criterion:          8.798637504793927e-6

 Number of observations:                  12

 Parameter estimates
───────────────────────────────────────
      Estimate   Std.Error  t-statistic
───────────────────────────────────────
Vm  212.684     6.94715        30.6145
K     0.064121  0.00828092      7.74322
───────────────────────────────────────
```
