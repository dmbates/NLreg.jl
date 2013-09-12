abstract PLregMod{T<:FP}

mm(m::PLregMod) = m.MM
mmd(m::PLregMod) = m.MMD
size(m::PLregMod) = size(mmd(m))
size(m::PLregMod,d::Integer) = size(mmd(m),d)

type PLregFit{T<:FP}
    m::PLregMod{T}
    y::Vector{T}
    resid::Vector{T}
    c::Vector{T}
    pars::Vector{T}
    incr::Vector{T}
    vs::Matrix{T}
    tr::Matrix{T}
    B::Matrix{T}
end
function PLregFit{T<:FP}(m::PLregMod{T},y::Vector{T})
    (n,nl,nnl) = size(m); length(y) == n || error("Dimension mismatch")
    PLregFit(m, y, similar(y), similar(y), Array(T,nl + nnl),
             Array(T,nnl), Array(T,n,nl), Array(T,nl,nl), Array(T,n,nnl))
end
PLregFit{T<:FP}(m::PLregMod{T},y::DataArray{T,1}) = PLregFit(m,vector(c))
PLregFit{T<:Integer}(m::PLregMod{T},y::DataArray{T,1}) = PLregFit(m,convert(Vector{T},vector(c)))

function deviance{T<:FP}(pl::PLregFit{T},nlp::Vector{T})
    m = pl.m; (n,nl,nnl) = size(m);
    lin = 1:nl; nonlin = nl + (1:nnl)
    vs = pl.vs; y = pl.y; resid = pl.resid; c = copy!(pl.c,y); pars = pl.pars
    copy!(sub(pars, nonlin), nlp)                   # record nl pars
    pl.tr = tr = qrfact!(copy!(vs,newpar(m,nlp))).T # update and factor mm
    copy!(resid,gemqrt!('L','T',vs,tr,c))           # c = Q'y
    trsv!('U','N','N',sub(vs,lin,lin),copy!(sub(pars,lin),sub(c,lin))) # lin. pars
    fill!(sub(resid, lin), zero(T))
    gemqrt!('L','N',vs,tr,resid)        # residuals
    sqsum(resid)
end

function gpinc(pl::PLregFit)
    m = pl.m; (n,nl,nnl) = size(m); Aphi = mmd(m); B = pl.B; r = pl.resid
    lin = 1:nl; lpars = sub(pl.pars,lin)
    for k in 1:nnl gemv!('N',1.,sub(Aphi,:,:,k),lpars,0.,sub(B,:,k)) end
    gemqrt!('L','T',pl.vs,pl.tr,B)
    for k in 1:nnl gemv!('T',1.,sub(Aphi,:,:,k),r,0.,sub(B,lin,k)) end
    trsm!('L','U','N','N',1.0,sub(pl.vs,lin,lin),sub(B,lin,:))
    rhs = copy(pl.c); fill!(sub(rhs,lin),zero(T)); pl.incr = B\rhs
    pl
end
    
