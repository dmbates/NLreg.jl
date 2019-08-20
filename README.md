# Nonlinear regression models

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dmbates.github.io/NLreg.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dmbates.github.io/NLreg.jl/dev)
[![Build Status](https://travis-ci.com/dmbates/NLreg.jl.svg?branch=master)](https://travis-ci.com/dmbates/NLreg.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/dmbates/NLreg.jl?svg=true)](https://ci.appveyor.com/project/dmbates/NLreg-jl)
[![Codecov](https://codecov.io/gh/dmbates/NLreg.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dmbates/NLreg.jl)

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
