abstract NLregMod{T<:FP}

size(m::NLregMod) = size(m.tgrad)
size(m::NLregMod,args...) = size(m.tgrad,args...)
## default methods for all NLregMod objects
residuals(m::NLregMod) = m.resid
model_response(m::NLregMod) = m.y

type NonlinearLS{T<:FP} <: RegressionModel # nonlinear least squares fits
    m::NLregMod{T}
    pars::Vector{T}
    incr::Vector{T}
    ch::Cholesky{T}
    rss::T      # residual sum of squares at last successful iteration
    tolsqr::T   # squared tolerance for orthogonality criterion
    minfac::T
    mxiter::Int
    fit::Bool
end
function NonlinearLS{T<:FP}(m::NLregMod{T},init::Vector{T})
    p,n = size(m); length(init) == p || error("Dimension mismatch")
    rss = updtmu!(m, init); tg = m.tgrad
    NonlinearLS(m, init, zeros(T,p), cholfact(eye(p)), rss, 1e-8, 0.5^10, 1000, false)
end
NonlinearLS{T<:FP}(m::NLregMod{T}) = NonlinearLS(m, initpars(m))

## returns a copy of the current parameter values
coef(nl::NonlinearLS) = copy(nl.pars)

## returns the coefficient table
function coeftable(nl::NonlinearLS)
    pp = coef(nl); se = stderr(nl); tt = pp ./ se
    CoefTable (DataFrame({pp, se, tt, ccdf(FDist(1, df_residual(nl)), tt .* tt)},
                         ["Estimate","Std.Error","t value", "Pr(>|t|)"]),
               pnames(nl), 4)
end

deviance(nl::NonlinearLS) = nl.rss

df_residual(nl::NonlinearLS) = ((p,n) = size(nl);n-p)

function gnfit(nl::NonlinearLS,verbose::Bool=false) # Gauss-Newton nonlinear least squares
    if !nl.fit
        m = nl.m; pars = nl.pars; incr = nl.incr; minf = nl.minfac; cvg = 2(tol = nl.tolsqr)
        r = m.resid; tg = m.tgrad; ch = nl.ch; nl.rss = rss = updtmu!(m,pars); UL = ch.UL
        for i in 1:nl.mxiter
            ## Create the Cholesky factor of tg * tg' in place
            _,info = potrf!('U',syrk!('U','N',1.,tg,0.,UL))
            info == 0 || error("Singular gradient matrix at pars = $(pars')")
            ## solve in place for the Gauss-Newton increment - done in two stages
            ## to be able to evaluate the orthogonality convergence criterion
            cvg = sumsq(trsv!('U','T','N',UL,gemv!('N',1.,tg,r,0.,incr)))/rss
            if verbose
                print("Iteration:",lpad(string(i),3),", rss = "); showcompact(rss)
                print(", cvg = "); showcompact(cvg); print(" at "); showcompact(pars)
                println()
            end
            trsv!('U','N','N',UL,incr)
            if verbose
                print("   Incr: ")
                showcompact(incr)
            end
            f = 1.
            while true
                f >= minf || error("Failure to reduce rss at $(pars') with incr = $(incr') and minfac = $minf")
                rss = updtmu!(nl.m, pars + f * incr)
                if verbose
                    print("  f = ",f,", rss = "); showcompact(rss); println()
                end
                rss < nl.rss && break
                f *= 0.5                    # step-halving
            end
            cvg < tol && break
            pars += f * incr
            nl.rss = rss
        end
        verbose && println()
        copy!(nl.pars,pars)
        cvg < tol || error("failure to converge in $(nl.mxiter) iterations")
        nl.fit = true
    end
    nl
end

nobs(nl::NonlinearLS) = size(nl,2)
    
pnames(nl::NonlinearLS) = pnames(nl.m)

function show{T<:FP}(io::IO, nl::NonlinearLS{T})
    gnfit(nl)
    p,n = size(nl)
    s2 = deviance(nl)/convert(T,n-p)
    println(io, "Nonlinear least squares fit to ", n, " observations")
## Add a model or modelformula specification in here
    println(io); show(io, coeftable(nl)); println(io)
    print(io,"Residual sum of squares at estimates: "); showcompact(io,nl.rss); println(io)
    print(io,"Residual standard error = ");showcompact(io,sqrt(s2));
    print(io, " on $(n-p) degrees of freedom")
end
    
size(nl::NonlinearLS) = size(nl.m)
size(nl::NonlinearLS,args...) = size(nl.m,args...)

stderr(nl::NonlinearLS) = sqrt(diag(vcov(nl)))
    
function vcov{T<:FP}(nl::NonlinearLS{T})
    p,n = size(nl)
    deviance(nl)/convert(T,n-p) * symmetrize!(potri!('U', copy(nl.ch.UL))[1], 'U')
end
