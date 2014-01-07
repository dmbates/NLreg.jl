abstract PLregMod{T<:FP} <: NLregMod{T}

function updtMM!(m::PLregMod,nlpars::StridedVector)
    x = m.x; MMD=m.MMD; nnl,nl,n = size(MMD); tg = sub(m.tgrad,1:nl,:)
    for i in 1:length(m.y)
        m.mmf(nlpars,sub(x,:,i),sub(tg,:,i),sub(MMD,:,:,i))
    end
    tg
end

mmd(m::PLregMod) = m.MMD

function updtmu!(m::PLregMod,pars::Vector)
    x = m.x; tg = m.tgrad; MMD = m.MMD
    nnl,nl,n = size(m); lind = 1:nl; nlind = nl + (1:nnl)
    nlpars = sub(pars,nlind); lpars = sub(pars,lind)
    for i in 1:n
        m.mmf(nlpars,sub(x,:,i),sub(tg,lind,i),sub(MMD,:,:,i))
        BLAS.gemv!('T',1.,sub(MMD,:,:,i),lpars,0.,sub(tg,nlind,i))
    end
    sumsq(map!(Subtract(),m.resid,m.y,BLAS.gemv!('T',1.,sub(tg,lind,:),lpars,0.,m.mu)))
end

size(pl::PLregMod) = size(pl.MMD)
size(pl::PLregMod,args...) = size(pl.MMD,args...)

type PLinearLS{T<:FP}
    m::PLregMod{T}
    qr::QR{T}
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
    qr = QR(updtMM!(m,nlpars)')
    pars = [qr\m.y,nlpars]
    PLinearLS(m, qr, pars, zeros(T,nnl), Array(T,n,nnl), updtmu!(m,pars),
              convert(T,1e-8), convert(T,0.5^10), 1000, false)
end
function PLinearLS{T<:FP}(m::PLregMod{T})
    nnl,nl,n = size(m)
    PLinearLS(m,initpars(m)[nl + (1:nnl)])
end

function deviance{T<:FP}(pl::PLinearLS{T},nlp::Vector{T})
    m = pl.m; nnl,nl,n = size(m); pars = pl.pars
    copy!(sub(pars,nl + (1:nnl)), nlp)   # record nl pars
    pl.qr = qr = QR(updtMM!(m,nlp)') # update and decompose lin pars model matrix
    copy!(sub(pars,1:nl),qr\m.y)     # conditionally optimal linear pars
    pl.rss = updtmu!(m,pars)         # update mu and evaluate deviance
end
deviance(pl::PLinearLS) = ((nnl,nl,n) = size(pl.m);deviance(pl,pl.pars[nl + (1:nnl)]))

function gpinc{T<:FP}(pl::PLinearLS{T})
    m = pl.m; nnl,nl,n = size(m); Aphi = m.MMD; B = pl.B; r = pl.m.resid
    lin = 1:nl; lpars = pl.pars[lin]
    for k in 1:nnl, i in 1:n
        B[i,k] = dot(lpars,Aphi[k,:,i])
    end
    LAPACK.gemqrt!('L','T',pl.qr.vs,pl.qr.T,B)
    for j in 1:nnl, i in 1:nl
        B[i,j] = dot(vec(Aphi[i,j,:]),r)
    end
    BLAS.trsm!('L','U','N','N',1.0,pl.qr[:R],sub(B,lin,:))
    rhs = pl.qr[:Q]'*m.y; for i in 1:nl rhs[i] = zero(T) end
    st = QR(B); sc = (st[:Q]'*rhs)[1:nnl]
    BLAS.trsv!('U','N','N',sub(st.vs,1:nnl,:),copy!(pl.incr,sc))
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
