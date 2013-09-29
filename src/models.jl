immutable MicMen{T<:FP} <: PLregMod{T}
    x::Vector{T}
    y::Vector{T}
    mu::Vector{T}
    resid::Vector{T}
    tgrad::Matrix{T}
    MM::Matrix{T}
    MMD::Array{T,3}
    MtM::Matrix{T}
end
function MicMen{T<:FP}(x::Vector{T},y::Vector{T})
    n = length(x); length(y) == n || error("Dimension mismatch")
    MicMen(x,y,similar(x),similar(x),Array(T,(2,n)),ones(T,(1,n)),zeros(T,(1,1,n)),zeros(T,(1,1)))
end
MicMen{T<:FP}(x::DataArray{T,1},y::DataArray{T,1}) = MicMen(vector(x),vector(y))
function MicMenf!{T<:FP}(K::T,xi::T,MM::UnsafeVectorView{T},MMD::UnsafeMatrixView{T})
    denom = K + xi
    MMD[1,1] = -(MM[1] =  xi/denom)/denom
end

pnames(m::MicMen) = ["Vm", "K"]
size(m::MicMen) = (length(m.x),1,1)

function updtMM!{T<:FP}(m::MicMen{T},K::T)
    x = m.x; MM = m.MM; MMD = m.MMD
    for i in 1:length(x)
        MicMenf!(K,x[i],unsafe_view(MM,:,i),unsafe_view(MMD,:,:,i))
    end
    tg
end
function updtmu!{T<:FP}(m::MicMen{T},K::T)
    Vm = Array(T,1); MtM = m.MtM; MM = m.MM; mu = m.mu
    _,info = potrf!('U',syrk!('U','N',one(T),m.MM,zero(T),MtM))
    info == 0 || error("Singular model matrix at K = $K")
    potrs!('U',MtM,Vm)
    sqdiffsum!(resid,resid,y,gemv!('T',one(T),MM,Vm,zero(T),mu),1)
end

immutable AsympReg{T<:FP} <: PLregMod{T}
    x::Vector{T}
end
AsympReg{T<:FP}(c::DataArray{T,1}) = AsympReg(vector(c))
AsympReg{T<:Integer}(c::DataArray{T,1}) = AsympReg(float(vector(c)))

pnames(m::AsympReg) = ["Asym","R0","rc"]

function updtMM!{T<:FP}(m::AsympReg{T},rc::T,MM::Matrix{T},MMD::Array{T,3})
    x = m.x
    for i in 1:length(x)
        xi = x[i]
        MMD[i,2,1] = -xi*(MM[i,2] = exp(-rc*xi))
    end
    MM
end
updtMM!{T<:FP}(m::AsympReg{T},nlp::Vector{T},MM::Matrix{T},MMD::Array{T,3}) = updtMM!(m,nlp[1],MM,MMD)

immutable Exp1{T<:FP} <: PLregMod{T}
    t::Vector{T}
end
Exp1{T<:FP}(t::DataArray{T,1}) = Exp1(vector(t))
Exp1{T<:Integer}(t::DataArray{T,1}) = Exp1(float(vector(t)))
pnames(m::Exp1) = ["V","ke"]
size(m::Exp1) = (length(m.x),1,1)
updt1!{T<:FP}(m::Exp1,t::T,ke::T,mm::StridedVector{T},mmd::StridedMatrix{T}) = mmd[1,1] = -t*(mm[1] = exp(-ke*t))

function updtMM!{T<:Float64}(m::Exp1{T},ke::T,MM::Matrix{T},MMD::Array{T,3})
    t=m.t; for i in 1:length(t) updt1!(m,t[i],ke,sub(MM,:,i),sub(MMD,:,:,i)) end
    MM
end
function updtMM!{T<:Float64}(m::Exp1{T},nlp::Vector{T},MM::Matrix{T},MMD::Array{T,3})
    t=m.t; ke = nlp[1]
    for i in 1:length(t) updt1!(m,t[i],ke,sub(MM,:,i),sub(MMD,:,:,i)) end
    MM
end
function updtMM!{T<:Float64}(m::Exp1{T},nlp::Matrix{T},MM::Matrix{T},MMD::Array{T,3})
    t=m.t; for i in 1:length(t) updt1!(m,t[i],nlp[i,1],sub(MM,:,i),sub(MMD,:,:,i)) end
    MM
end

immutable logsd1{T<:FP} <: NLregMod{T}
    t::Vector{T}
    y::Vector{T}
    mu::Vector{T}
    resid::Vector{T}
    tgrad::Matrix{T}
end
function logsd1{T<:FP}(t::Vector{T},y::Vector{T})
    n = length(t); length(y) == n || error("Dimension mismatch")
    logsd1(t,y,similar(t),similar(t),Array(T,(2,n)))
end
logsd1{T<:FP}(t::DataArray{T,1},y::DataArray{T,1}) = logsd1(vector(t),vector(y))
function logsd1(f::Formula,dat::AbstractDataFrame)
    mf = ModelFrame(f,dat)
    mm = ModelMatrix(mf)
    logsd1(mm.m[:,2],model_response(mf))
end
logsd1(ex::Expr,dat::AbstractDataFrame) = logsd1(Formula(ex),dat)

function logsd1f{T<:FP}(V::T,K::T,ti::T,mu::UnsafeVectorView{T},tg::UnsafeVectorView{T})
    tg[2] = -ti*K*(tg[1] = mu[1] = mm = V*exp(-K*ti))
    mm
end

function updtmu!{T<:FP}(m::logsd1{T}, pars::Vector{T})
    V = exp(pars[1]); K = exp(pars[2]); t = m.t; mu = m.mu; tgrad = m.tgrad;
    y = m.y; r = m.resid; rss = zero(T)
    for i in 1:length(t)
        ri = r[i] = y[i] - logsd1f(V,K,t[i],unsafe_view(mu,i:i),unsafe_view(tgrad,:,i))
        rss += abs2(ri)
    end
    rss
end
function updtmu!{T<:FP}(m::logsd1{T}, p::Matrix{T})
    t = m.t; mu = m.mu; tgrad = m.tgrad; rss = zero(T)
    y = m.y; r = m.resid
    size(pars) == size(tgrad) || error("Dimension mismatch")
    for i in 1:length(t)
        ri = r[i] = y[i] - logsd1f(exp(p[1,i]),exp(p[2,i]),t[i],
                                   unsafe_view(mu,i:i),unsafe_view(tgrad,:,i))
        rss += abs2(ri)
    end
    rss
end
function updtmu!{T<:FP}(m::logsd1{T}, pars::Matrix{T}, inds::Vector)
    t = m.t; mu = m.mu; tgrad = m.tgrad; k,n = size(tgrad); rss = zero(T)
    length(inds) == n && size(pars,1) == k || error("Dimension mismatch")
    y = m.y; resid = m.resid; ii = 0; V = 0.; K = 0.
    for i in 1:n
        if ii != inds[i]
            ii = inds[i]; V = exp(pars[1,ii]); K = exp(pars[2,ii])
        end
        ri = resid[i] = y[i] - logsd1f(V,K,t[i],unsafe_view(mu,i:i),unsafe_view(tgrad,:,i))
        rss += abs2(ri)
    end
    rss
end

size(m::logsd1) = size(m.tgrad)
size(m::logsd1,args...) = size(m.tgrad,args...)

function initpars{T<:FP}(m::logsd1{T})
    (n = length(m.t)) < 2 && return [zero(T),-one(T)]
    cc = hcat(ones(n),m.t)\log(m.y)
    cc[2] < 0. ? [cc[1],log(-cc[2])] : [cc[1],-one(T)]
end

pnames(m::logsd1) = ["logV","logK"]
