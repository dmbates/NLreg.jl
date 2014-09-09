vcov(rm::RegressionModel) = scale(rm,true) * unscaledvcov(rm)

type NonlinearLS{T<:FP} <: RegressionModel # nonlinear least squares fits
    m::NLregModF{T}
    pars::Vector{T}
    incr::Vector{T}
    ch::Base.Cholesky{T}
    rss::T      # residual sum of squares at last successful iteration
    tolsqr::T   # squared tolerance for orthogonality criterion
    minfac::T
    mxiter::Int
    fit::Bool
end
function NonlinearLS{T<:FP}(m::NLregModF{T},init::Vector{T})
    p,n = size(m)
    if isa(m,PLregMod)
        nnl,nl,n = size(m)
        p = nl + nnl
    end
    length(init) == p || error("Dimension mismatch")
    rss = updtmu!(m, init); tg = m.tgrad
    NonlinearLS(m, init, zeros(T,p), cholfact(eye(p)), rss, 1e-8, 0.5^10, 1000, false)
end
NonlinearLS{T<:FP}(m::NLregModF{T}) = NonlinearLS(m, initpars(m))
 
## returns a copy of the current parameter values
coef(nl::NonlinearLS) = copy(nl.pars)

## returns the coefficient table
function coeftable(nl::NonlinearLS)
    pp = coef(nl); se = stderr(nl); tt = pp ./ se
    CoefTable(hcat(pp, se, tt, ccdf(FDist(1, df_residual(nl)), abs2(tt))),
              ["Estimate","Std.Error","t value", "Pr(>|t|)"],
              map(string,pnames(nl)[1:length(pp)]), 4)
end
 
deviance(nl::NonlinearLS) = nl.rss
df_residual(nl::NonlinearLS) = ((p,n) = size(nl);n-p)
function gnfit(nl::NonlinearLS,verbose::Bool=false) # Gauss-Newton nonlinear least squares
    if !nl.fit
        m = nl.m; pars = nl.pars; incr = nl.incr; minf = nl.minfac; cvg = 2(tol = nl.tolsqr)
        r = m.resid; tg = m.tgrad; ch = nl.ch; nl.rss = rss = updtmu!(m,pars); UL = ch.UL
        for i in 1:nl.mxiter
            ## Create the Cholesky factor of tg * tg' in place
            _,info = LAPACK.potrf!('U',BLAS.syrk!('U','N',1.,tg,0.,UL))
            info == 0 || error("Singular gradient matrix at pars = $(pars')")
            ## solve in place for the Gauss-Newton increment - done in two stages
            ## to be able to evaluate the orthogonality convergence criterion
            cvg = sumsq(BLAS.trsv!('U','T','N',UL,BLAS.gemv!('N',1.,tg,r,0.,incr)))/rss
            if verbose
                print("Iteration:",lpad(string(i),3),", rss = "); showcompact(rss)
                print(", cvg = "); showcompact(cvg); print(" at "); showcompact(pars)
                println()
            end
            BLAS.trsv!('U','N','N',UL,incr)
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
model(nl::NonlinearLS) = nl.m
nobs(nl::NonlinearLS) = size(nl,2)
pnames(nl::NonlinearLS) = pnames(model(nl))

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
size(nl::NonlinearLS) = size(tgrad(model(nl)))
size(nl::NonlinearLS,args...) = size(tgrad(model(nl)),args...)

type PLinearLS{T<:FP} <: RegressionModel
    m::PLregModF{T}
    qr::QRCompactWY{T}
    pars::Vector{T}
    incr::Vector{T}
    B::Matrix{T}
    rss::T      # residual sum of squares at last successful iteration
    tolsqr::T   # squared tolerance for orthogonality criterion
    minfac::T
    mxiter::Int
    fit::Bool
end
function PLinearLS{T<:FP}(m::PLregModF{T},nlpars::Vector{T})
    nnl,nl,n = size(m)
    length(nlpars) == nnl || throw DimensionMismatch("")
    qr = qrfact!(updtMM!(m,nlpars)')
    pars = [vec(qr\model_response(m)),nlpars]
    PLinearLS(m, qr, pars, zeros(T,nnl), Array(T,n,nnl), updtmu!(m,pars),
              convert(T,1e-8), convert(T,0.5^10), 1000, false)
end
PLinearLS(m::PLregModF) = PLinearLS(m,initpars(m))

pnames(pl::PLinearLS) = pnames(pl.m)

function deviance{T<:FP}(pl::PLinearLS{T},nlp::Vector{T})
    m = pl.m; nnl,nl,n = size(m.MMD); pars = pl.pars
    copy!(view(pars,nl + (1:nnl)), nlp)   # record nl pars
    pl.qr = qr = qrfact!(updtMM!(m,nlp)') # update and decompose lin pars model matrix
    copy!(view(pars,1:nl),qr\model_response(m))     # conditionally optimal linear pars
    pl.rss = updtmu!(m,pars)         # update mu and evaluate deviance
end
deviance(pl::PLinearLS) = pl.rss

function gpinc{T<:FP}(pl::PLinearLS{T})
    m = pl.m; nnl,nl,n = size(m); Aphi = mmjac(m); B = pl.B
    r = residuals(m); mqr = pl.qr
    lin = 1:nl; lpars = pl.pars[lin]
    for k in 1:nnl
        B[:,k] = reshape(Aphi[k,:,:],(nl,n))' * lpars
    end
    LAPACK.gemqrt!('L','T',mqr.factors,mqr.T,B)
    for j in 1:nnl, i in 1:nl
        B[i,j] = dot(vec(Aphi[j,i,:]),r)
    end
    BLAS.trsm!('L','U','N','N',1.0,mqr[:R],sub(B,lin,:))
    rhs = mqr[:Q]'*model_response(m); for i in 1:nl rhs[i] = zero(T) end
    st = qrfact(B); sc = (st[:Q]'*rhs)[1:nnl]
    BLAS.trsv!('U','N','N',sub(st.factors,1:nnl,:),copy!(pl.incr,sc))
    sumsq(sc)/pl.rss
end

function gpfit(pl::PLinearLS,verbose::Bool=false) # Golub-Pereyra variable projection algorithm
    if !pl.fit
        m = pl.m; pars = pl.pars; nnl,nl,n = size(m.MMD); nlin = nl + (1:nnl);
        minf = pl.minfac; cvg = 2(tol = pl.tolsqr); oldrss = pl.rss
        for i in 1:pl.mxiter
            cvg = gpinc(pl) # evaluate the increment and convergence criterion
            if verbose
                print("Iteration:",lpad(string(i),3),", rss = "); showcompact(oldrss)
                print(", cvg = "); showcompact(cvg); print(" at "); showcompact(pl.pars)
                println(); print("   Incr: "); showcompact(pl.incr)
            end
            f = 1.
            while true
                f >= minf || error("Failure to reduce rss at $(pars') with incr = $(pl.incr') and minfac = $minf")
                deviance(pl, pl.pars[nlin] + f * pl.incr)
                if verbose
                    print("  f = ",f,", rss = "); showcompact(pl.rss); println()
                end
                pl.rss < oldrss && break
                f *= 0.5                    # step-halving
            end
            cvg < tol && break
            oldrss = pl.rss
        end
        verbose && println()
        cvg < tol || error("failure to converge in $(pl.mxiter) iterations")
        pl.fit = true
    end
    pl
end

## returns a copy of the current parameter values
coef(pl::PLinearLS) = copy(pl.pars)

## returns the coefficient table
function coeftable(pl::PLinearLS)
    pp = coef(pl); se = stderr(pl); tt = pp ./ se
    CoefTable(hcat(pp, se, tt, ccdf(FDist(1, df_residual(pl)), abs2(tt))),
              ["Estimate","Std.Error","t value", "Pr(>|t|)"],
              map(string,pnames(pl)[1:length(pp)]), 4)
end

df_residual(pl::PLinearLS) = nobs(pl) - npars(pl)

fit(m::PLregModF,verbose=false) = gpfit(PLinearLS(m),verbose)

model_response(pl::PLinearLS) = model_response(pl.m)

npars(pl::PLinearLS) = npars(pl.m)

function scale(pl::PLinearLS,sqr=false)
    scsq = deviance(pl)/df_residual(pl)
    sqr ? scsq : sqrt(scsq)
end

residuals(pl::PLinearLS) = residuals(pl.m)

function show{T<:FP}(io::IO, pl::PLinearLS{T})
    gpfit(pl)
    println(io, "Nonlinear least squares fit to $(nobs(pl)) observations")
## Add a model or modelformula specification in here
    println(io); show(io, coeftable(pl)); println(io)
    print(io,"Residual sum of squares at estimates: "); showcompact(io,pl.rss); println(io)
    print(io,"Residual standard error = ");showcompact(io,scale(pl));
    print(io, " on $(df_residual(pl)) degrees of freedom")
end

size(pl::PLinearLS) = size(pl.m.MMD)

size(pl::PLinearLS,args...) = size(pl.m.MMD,args...)

vcov(pl::PLinearLS) = scale(pl,true) * unscaledvcov(pl.m)

