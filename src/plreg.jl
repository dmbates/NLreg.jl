abstract PLregMod{T<:FP} <: NLregMod{T}

function updtMM!(m::PLregMod,nlpars::StridedVector)
    x = m.x; MMD=m.MMD; nnl,nl,n = size(MMD); tg = sub(m.tgrad,1:nl,:)
    for i in 1:length(m.y)
        m.mmf(nlpars,sub(x,:,i),sub(tg,:,i),sub(MMD,:,:,i))
    end
    tg
end

function updtmu!(m::PLregMod,pars::Vector)
    x = m.x; tg = m.tgrad; MMD = m.MMD
    nl,nnl,n = size(m); lind = 1:nl; nlind = nl + (1:nnl)
    nlpars = sub(pars,nlind); lpars = sub(pars,lind)
    for i in 1:n
        m.mmf(nlpars,sub(x,:,i),sub(tg,lind,i),sub(MMD,:,:,i))
        gemv!('T',1.,sub(MMD,:,:,i),lpars,0.,sub(tg,nlind,i))
    end
    sumsq(map!(Subtract(),m.resid,m.y,gemv!('T',1.,sub(tg,lind,:),lpars,0.,m.mu)))
end

size(pl::PLregMod) = size(pl.MMD)
size(pl::PLregMod,args...) = size(pl.MMD,args...)

type PLinearLS{T<:FP}
    m::PLregMod{T}
    qr::QR{T}
    pars::Vector{T}
    incr::Vector{T}
    B::Matrix{T}
end
function PLinearLS{T<:FP}(m::PLregMod{T},nlpars::Vector{T})
    nl,nnl,n = size(m); length(nlpars) == nnl || error("Dimension mismatch")
    qr = QR(updtMM!(m,nlpars)')
    PLinearLS(m, qr, [qr\m.y,nlpars], zeros(T,nnl), Array(T,nl,nnl))
end
PLinearLS{T<:FP}(m::PLregMod{T},y::DataArray{T,1}) = PLinearLS(m,vector(c))
PLinearLS{T<:Integer}(m::PLregMod{T},y::DataArray{T,1}) = PLinearLS(m,convert(Vector{T},vector(c)))

function deviance{T<:FP}(pl::PLinearLS{T},nlp::Vector{T})
    m = pl.m; nl,nnl,n = size(m); lind = 1:nl; nlind = nl + (1:nnl)
    y = m.y; r = m.resid; pars = pl.pars
    copy!(sub(pars,nlind), nlp)         # record nl pars
    updtMM!(m,nlp)       # update the model matrix for the linear pars
    pl.qr = qr = QR(sub(m.tgrad,lind,:)')
    copy!(sub(pars,lind),qr\y)
    updtmu!(m,pars)
end

function gpinc{T<:FP}(pl::PLinearLS{T})
    m = pl.m; (n,nl,nnl) = size(m); Aphi = mmd(m); B = pl.B; r = pl.resid
    lin = 1:nl; lpars = sub(pl.pars,lin)
    for k in 1:nnl gemv!('N',1.,sub(Aphi,:,:,k),lpars,0.,sub(B,:,k)) end
    gemqrt!('L','T',pl.vs,pl.tr,B)
    for k in 1:nnl gemv!('T',1.,sub(Aphi,:,:,k),r,0.,sub(B,lin,k)) end
    trsm!('L','U','N','N',1.0,sub(pl.vs,lin,lin),sub(B,lin,:))
    rhs = copy(pl.c); fill!(sub(rhs,lin),zero(T)); pl.incr = B\rhs
    pl
end
    
