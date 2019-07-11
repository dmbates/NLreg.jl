module NLreg

    using DataFrames, LinearAlgebra
    using DiffResults: JacobianResult, MutableDiffResult
    using ForwardDiff: JacobianConfig, jacobian!
    using StatsBase: CoefTable, StatisticalModel
    import StatsBase: coef, coefnames, coeftable, confint, deviance, dof, dof_residual
    import StatsBase: fit, fit!, fitted, islinear, loglikelihood, mss, nobs, params
    import StatsBase: residuals, response, rss, stderror, vcov

    export
        NLregModel,
        coef,
        coefnames,
        coeftable,
        confint,
        deviance,
        dof,
        dof_residual,
        fit!,
        fitted,
        islinear,
        loglikelihood,
        mss,
        nobs,
        params,
        residuals,
        response,
        rss,
        stderror,
        vcov

    include("utilities.jl")
    include("nonlinreg.jl")

end # module
