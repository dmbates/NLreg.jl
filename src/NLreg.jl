module NLreg

using LinearAlgebra, StatsBase, Tables, Zygote

export
    NLregModel,
    coef,
    coefnames,
    coeftable,
    confint,
    deviance,
    dof,
    dof_residual,
    fit,
    fit!,
    fitted,
    islinear,
    loglikelihood,
    mss,
    nobs,
    residuals,
    response,
    rss,
    stderror,
    vcov

include("nlregfit.jl")

end # module
