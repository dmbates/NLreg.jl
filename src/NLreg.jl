module NLreg

using LinearAlgebra, StatsBase, Tables, Zygote

export
    NLregModel,
    coef,
    coefnames,
    coeftable,
    confint,
    deviance,
    dispersion,
    dof,
    dof_residual,
    fit,
    fit!,
    fitted,
    islinear,
    loglikelihood,
    mss,
    nobs,
    objective,
    residuals,
    response,
    rss,
    stderror,
    vcov

include("nlreg.jl")

end # module
