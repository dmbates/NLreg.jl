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
    fit!,
    fitted,
    islinear,
    loglikelihood,
    mss,
    nobs,
    objective,
    params,
    pnls!,
    residuals,
    response,
    rss,
    stderror,
    updateL!,
    updateμ!,
    vcov

include("nlreg.jl")

end # module
