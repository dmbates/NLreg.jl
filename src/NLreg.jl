module NLreg

    using DataFrames, NumericExtensions
    using Base.LinAlg.BLAS: gemv!, trsm!, trsv!
    using Base.LinAlg.LAPACK: gemqrt!,geqrt3!, potri!

    import Base: size, show
    import Distributions: fit
    import GLM: deviance
    import NumericExtensions: evaluate, result_type

    export                              # types
        AsympReg,                       # Asymptotic regression model
        MicMen,                         # Michaelis-Menten model
        PLregFit,
        PLregMod,

        gpinc

    typealias FP FloatingPoint

    include("plreg.jl")
    include("models.jl")

end # module
