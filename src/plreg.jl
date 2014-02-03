abstract PLregMod{T<:FP} <: NLregMod{T}

## default methods for all PLregMod objects
model_response(m::PLregMod) = m.y
residuals(m::PLregMod) = m.resid
size(pl::PLregMod) = size(pl.MMD)
size(pl::PLregMod,args...) = size(pl.MMD,args...)

## update the (transposed) model matrix for the linear parameters and the gradient MMD
function updtMM!(m::PLregMod,nlpars)
    x = m.x; MMD=m.MMD; tg = m.tgrad
    for i in 1:size(x,2)
        m.mmf(nlpars,view(x,:,i),view(tg,:,i),view(MMD,:,:,i))
    end
    tg[1:size(m,2),:]
end

## update mu and resid from full parameter vector returning rss
function updtmu!(m::PLregMod,pars::Vector)
    x = m.x; tg = m.tgrad; MMD = m.MMD; mu = m.mu; mmf = m.mmf
    nnl,nl,n = size(m); lind = 1:nl; nlind = nl + (1:nnl)
    nlpars = pars[nlind]; lpars = pars[lind]
    for i in 1:n
        mmf(nlpars,view(x,:,i),view(tg,:,i),view(MMD,:,:,i))
        tg[nlind,i] = MMD[:,:,i] * lpars
        mu[i] = dot(tg[lind,i],lpars)
    end
    sumsq(map!(Subtract(),m.resid,m.y,mu))
end

type PLinearLS{T<:FP} <: RegressionModel
    m::PLregMod{T}
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
function PLinearLS{T<:FP}(m::PLregMod{T},nlpars::Vector{T})
    nnl,nl,n = size(m); length(nlpars) == nnl || error("Dimension mismatch")
    qr = qrfact!(updtMM!(m,nlpars)')
    pars = [vec(qr\m.y),nlpars]
    PLinearLS(m, qr, pars, zeros(T,nnl), Array(T,n,nnl), updtmu!(m,pars),
              convert(T,1e-8), convert(T,0.5^10), 1000, false)
end
PLinearLS(m::PLregMod,nlp::Number) = PLinearLS(m,[convert(eltype(m.y),nlp)])
PLinearLS(m::PLregMod) = PLinearLS(m,initpars(m))

pnames(pl::PLinearLS) = pnames(pl.m)

function deviance{T<:FP}(pl::PLinearLS{T},nlp::Vector{T})
    m = pl.m; nnl,nl,n = size(m); pars = pl.pars
    copy!(sub(pars,nl + (1:nnl)), nlp)   # record nl pars
    pl.qr = qr = qrfact!(updtMM!(m,nlp)') # update and decompose lin pars model matrix
    copy!(sub(pars,1:nl),qr\m.y)     # conditionally optimal linear pars
    pl.rss = updtmu!(m,pars)         # update mu and evaluate deviance
end
deviance(pl::PLinearLS) = pl.rss

function gpinc{T<:FP}(pl::PLinearLS{T})
    m = pl.m; nnl,nl,n = size(m); Aphi = m.MMD; B = pl.B; r = m.resid; mqr = pl.qr
    lin = 1:nl; lpars = pl.pars[lin]
    for k in 1:nnl
        B[:,k] = reshape(Aphi[k,:,:],(nl,n))' * lpars
    end
    LAPACK.gemqrt!('L','T',mqr.factors,mqr.T,B)
    for j in 1:nnl, i in 1:nl
        B[i,j] = dot(vec(Aphi[j,i,:]),r)
    end
    BLAS.trsm!('L','U','N','N',1.0,mqr[:R],sub(B,lin,:))
    rhs = mqr[:Q]'*m.y; for i in 1:nl rhs[i] = zero(T) end
    st = qrfact(B); sc = (st[:Q]'*rhs)[1:nnl]
    BLAS.trsv!('U','N','N',sub(st.factors,1:nnl,:),copy!(pl.incr,sc))
    sumsq(sc)/pl.rss
end

function gpfit(pl::PLinearLS,verbose::Bool=false) # Golub-Pereyra variable projection algorithm
    if !pl.fit
        m = pl.m; pars = pl.pars; nnl,nl,n = size(m); nlin = nl + (1:nnl);
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
    CoefTable (DataFrame({pp, se, tt, ccdf(FDist(1, df_residual(pl)), tt .* tt)},
                         ["Estimate","Std.Error","t value", "Pr(>|t|)"]),
               pnames(pl), 4)
end

size(pl::PLinearLS) = size(pl.m.MMD)

size(pl::PLinearLS,args...) = size(pl.m.MMD,args...)
    
function vcov{T<:FP}(pl::PLinearLS{T})
    nnl,nl,n = size(pl); tg = pl.m.tgrad
    deviance(pl)/convert(T,n-(nl+nnl)) * inv(cholfact(tg * tg'))
end

df_residual(pl::PLinearLS) = ((nnl,nl,n) = size(pl);n-(nl+nnl))

model_response(pl::PLinearLS) = pl.m.y

residuals(pl::PLinearLS) = pl.m.resid

function show{T<:FP}(io::IO, pl::PLinearLS{T})
    gpfit(pl)
    nnl,nl,n = size(pl); p = nl + nnl
    s2 = deviance(pl)/convert(T,n-p)
    println(io, "Nonlinear least squares fit to ", n, " observations")
## Add a model or modelformula specification in here
    println(io); show(io, coeftable(pl)); println(io)
    print(io,"Residual sum of squares at estimates: "); showcompact(io,pl.rss); println(io)
    print(io,"Residual standard error = ");showcompact(io,sqrt(s2));
    print(io, " on $(n-p) degrees of freedom")
end

fit(m::PLregMod,verbose::Bool) = gpfit(PLinearLS(m),verbose)
fit(m::PLregMod) = fit(m,false)
