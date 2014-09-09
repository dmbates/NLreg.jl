module NLreg

    using DataArrays, DataFrames, NumericExtensions, NLopt, Base.Grisu
    using Base.LinAlg.CHOLMOD: CholmodSparse!, CholmodSparse, CholmodFactor
    using StatsBase: CoefTable, StatisticalModel, RegressionModel, range
    using Distributions: FDist, ccdf
    using Base.LinAlg.QRCompactWY

    import DataFrames: model_response
    import Distributions: fit
    import GLM: formula                 # move this to StatsBase?
    import StatsBase: coef, coeftable, confint, deviance, loglikelihood, nobs,
               stderr, vcov, predict, residuals, rle

    export     # Partially linear models
        AsympOff,     # Asymptotic regression expressed as an offset
        AsympOrig,    # Asymptotic regression through origin
        AsympReg,     # Asymptotic regression model
        Biexp,        # Bi-exponential model
        BolusSD1,     # 1-compartment, single bolus dose model
        Chwirut,      # Model used in the NIST nonlinear regression examples
        Gompertz,     # Gompertz growth model
        Logis3P,      # 3-parameter logistic
        Logis4P,      # 4-parameter logistic
        MicMen,       # Michaelis-Menten model
              # types for models functions and model fits
        CompositeModF,   # Nonlinear regression model function with parameter transformation
        CompositePLModF, # Partially linear regression model function with parameter trans
        NLregModF,       # Nonlinear regression model function
        NonlinearLS,     # Nonlinear least squares model
        PLinearLS,       # Partially linear regression model
        PLregModF,       # Partially linear regression model function
        NLMM,            # Nonlinear mixed-effects model
        SimpleNLMM,      # Simple population nonlinear mixed-effects model
              # parameter transformations
        Dtrans,          # diagonal parameter transformation
        IdTr,            # scalar identity transformation
        LogTr,           # scalar logarithmic transformation
        Ptrans,          # abstract parameter transformation
        Strans,          # scalar parameter transformation
              # Full, nonlinear regression models
        Logsd1,       # 1-compartment, single bolus dose model with logged parameters

        fit,          # fit the model
        incr!,        # increment the spherical random effects
        initpars,     # create initial values for the parameters
        lowerbd,      # lower bounds on NLMM parameters
        npars,        # total number of parameters in an NLregMod
        pnames,       # names of parameters in a model
        pnls!,        # penalized nonlinear least squares fit
        prss!,        # penalized rss for b = lambda * (u + fac*delu)
        setpars!,     # set new parameter values (beta + theta) in an NLMM
        theta,        # extract covariance parameters
        theta!,       # set covariance parameters
        updtMM!,      # update the model matrix in a PLregMod
        updtL!,       # update L and solve for delu
        updtmu!       # update mu and tgrad

    typealias FP FloatingPoint

    include("nonlinreg.jl")
    include("plreg.jl")
    include("models.jl")
    include("nlmm.jl")
    include("simplenlmm.jl")
    include("ptrans.jl")
    include("compositemodel.jl")

end # module
