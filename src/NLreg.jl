module NLreg

    using DataFrames, LinearAlgebra
    using DiffResults: JacobianResult, MutableDiffResult
    using ForwardDiff: JacobianConfig, jacobian!
    using StatsBase: CoefTable, StatisticalModel
    import StatsBase: coef, coefnames, coeftable, confint, deviance, dof, dof_residual
    import StatsBase: fit, fit!, fitted, islinear, loglikelihood, mss, nobs, params
    import StatsBase: residuals, response, rss, stderror, vcov

    export
        NLmixedModel,
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

    include("utilities.jl")
    include("nonlinreg.jl")
    include("nonlinmixed.jl")

end # module
