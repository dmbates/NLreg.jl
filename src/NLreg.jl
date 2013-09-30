module NLreg

    using DataFrames, NumericExtensions
    using Base.LinAlg.BLAS: gemv!, trmm!, trsm!, trsv!, syrk!
    using Base.LinAlg.LAPACK: gemqrt!,geqrt3!, potri!, potrf!, potrs!
    using Base.LinAlg.CHOLMOD: CholmodSparse!, CholmodSparse, CholmodFactor
    using Stats: StatisticalModel

    import Base: Triangular, copy, size, show
    import Distributions: fit
    import GLM: deviance
    import Stats: coef, coeftable, confint, stderr, vcov, residuals, model_response, predict
    import NumericExtensions: evaluate, result_type

    export                              # types
        AsympReg,                       # Asymptotic regression model
        MicMen,                         # Michaelis-Menten model
        NLregMod,
        NonlinearLS,
        PLregFit,
        PLregMod,
        SimplePopPK,
        logsd1,

        gnfit,
        gpinc,
        initpars,
        pnames,
        pnls!,
        prss!,
        updtmu!,
        updtMM!

    typealias FP FloatingPoint

    include("nlreg.jl")
    include("plreg.jl")
    include("models.jl")
    include("poppk.jl")

end # module
