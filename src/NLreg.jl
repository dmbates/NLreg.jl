module NLreg

    using DataFrames, LinearAlgebra
    using DiffResults: JacobianResult, MutableDiffResult
    using ForwardDiff: JacobianConfig, jacobian!
    using StatsBase: StatisticalModel
    import StatsBase: fit, fit!

    export
        NLregModel,
        fit!

    include("nonlinreg.jl")

end # module
