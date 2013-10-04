module NLreg

    using DataFrames, NumericExtensions
    using Base.LinAlg.BLAS: gemv!, trmm!, trsm!, trsv!, syrk!
    using Base.LinAlg.LAPACK: gemqrt!,geqrt3!, potri!, potrf!, potrs!
    using Base.LinAlg.CHOLMOD: CholmodSparse!, CholmodSparse, CholmodFactor
    using Stats: StatisticalModel, RegressionModel

    import Base: Triangular, copy, size, show
    import Distributions: fit
    import GLM: deviance
    import NumericExtensions: evaluate, result_type
    import Stats: coef, coeftable, confint, loglikelihood, nobs, stderr, vcov,
                  model_response, predict, residuals, rle

    export                              # types
        AsympReg,      # Asymptotic regression model
        MicMen,        # Michaelis-Menten model
        NLregMod,      # Nonlinear regression model abstract type
        NonlinearLS,   # Nonlinear least squares fit
        PLregFit,      # Partially linear regression model fit
        PLregMod,      # Partially linear regression model
        SimpleNLMM,    # Simple population nonlinear mixed-effects model
        Logsd1,        # 1-compartment, single bolus dose model with logged parameters

        gnfit,         # nonlinear least squares by Gauss-Newton
        gpinc,         # Golub-Pereyra increment
        incr!,         # increment the spherical random effects
        initpars,      # create initial values for the parameters
        pnames,        # names of parameters in a model
        pnls!,         # penalized nonlinear least squares fit
        prss!,         # penalized rss for b = lambda * (u + fac*delu)
        updtMM!,       # update the model matrix in a PLregMod
        updtL!,        # update L and solve for delu
        updtmu!        # update mu and tgrad

    typealias FP FloatingPoint

    include("nlreg.jl")
    include("plreg.jl")
    include("models.jl")
    include("poppk.jl")

end # module
