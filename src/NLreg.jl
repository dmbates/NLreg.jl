module NLreg

    using DataFrames, LinearAlgebra
    using DiffResults: JacobianResult, MutableDiffResult
    using ForwardDiff: JacobianConfig, jacobian!
    using StatsBase: StatisticalModel
    import StatsBase: fit, fit!, fitted, params, residuals

    export
        NLregModel,
        fit!,
        fitted,
        params,
        residuals

    include("utilities.jl")
    include("nonlinreg.jl")

end # module
