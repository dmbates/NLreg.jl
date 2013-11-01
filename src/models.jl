function updtmu!{T<:FP}(m::NLregMod{T}, pars::Vector{T})
    x = m.x; mu = m.mu; tgrad = m.tgrad;
    y = m.y; r = m.resid; rss = zero(T)
    for i in 1:length(y)
        mu[i] = m.f(pars,sub(x,:,i),sub(tgrad,:,i)) # pass subarrays by reference
        r[i] = y[i] - mu[i]
        rss += abs2(r[i])
    end
    rss
end
function updtmu!{T<:FP}(m::NLregMod{T}, pars::Matrix{T}, inds::Vector)
    x = m.x; mu = m.mu; tgrad = m.tgrad; k,n = size(tgrad); rss = zero(T)
    length(inds) == n && size(pars,1) == k || error("Dimension mismatch")
    y = m.y; r = m.resid; ii = 0
    for i in 1:n
        if ii != inds[i]
            ii = inds[i]
        end
        mu[i] = m.f(sub(pars,:,ii),sub(x,:,i),sub(tgrad,:,i))
        r[i] = y[i] - mu[i]
        rss += abs2(r[i])
    end
    rss
end
updtmu!{T<:FP}(m::NLregMod{T}, p::Matrix{T}) = updtmu!(m,p,1:length(m.y))

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
size(m::MicMen) = (2,length(m.x),1)

function updtMM!{T<:FP}(m::MicMen{T},K::T)
    x = m.x; MM = m.MM; MMD = m.MMD; MtM = m.MtM
    for i in 1:length(x)
        MicMenf!(K,x[i],unsafe_view(MM,:,i),unsafe_view(MMD,:,:,i))
    end
    MM
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

immutable Logsd1{T<:FP} <: NLregMod{T}
    x::Matrix{T}
    y::Vector{T}
    mu::Vector{T}
    resid::Vector{T}
    tgrad::Matrix{T}
    f::Function
end
function Logsd1{T<:FP}(t::Vector{T},y::Vector{T})
    n = length(y); length(t) == n || error("Dimension mismatch")
    Logsd1(reshape(t,(1,n)),y,similar(y),similar(y),Array(T,(2,n)),Logsd1f)
end
Logsd1{T<:FP}(t::DataArray{T,1},y::DataArray{T,1}) = Logsd1(vector(t),vector(y))
function Logsd1(f::Formula,dat::AbstractDataFrame)
    mf = ModelFrame(f,dat)
    mm = ModelMatrix(mf)
    Logsd1(mm.m[:,end],model_response(mf))
end
Logsd1(ex::Expr,dat::AbstractDataFrame) = Logsd1(Formula(ex),dat)

function Logsd1f{T<:FP}(V::T,K::T,ti::T,mu::UnsafeVectorView{T},tg::UnsafeVectorView{T})
    tg[2] = -ti*K*(tg[1] = mu[1] = mm = V*exp(-K*ti))
    mm
end

function initpars{T<:FP}(m::Logsd1{T})
    (n = length(m.t)) < 2 && return [zero(T),-one(T)]
    cc = hcat(ones(n),m.t)\log(m.y)
    cc[2] < 0. ? [cc[1],log(-cc[2])] : [cc[1],-one(T)]
end

pnames(m::Logsd1) = ["logV","logK"]

immutable BolusSD1{T<:FP} <: NLregMod{T}
    x::Matrix{T}
    y::Vector{T}
    mu::Vector{T}
    resid::Vector{T}
    tgrad::Matrix{T}
    f::Function
end
function BolusSD1(x::Matrix,y::Vector,mu::Vector,resid::Vector,tgrad::Matrix)
    BolusSD1(x,y,mu,resid,tgrad,BolusSD1f)
end
function BolusSD1{T<:FP}(t::Vector{T},y::Vector{T})
    n = length(y); length(t) == n || error("Dimension mismatch")
    BolusSD1(reshape(t,(1,n)),y,similar(y),similar(y),Array(T,(2,n)),BolusSD1f)
end
BolusSD1{T<:FP}(t::DataArray{T,1},y::DataArray{T,1}) = BolusSD1(vector(t),vector(y))
function BolusSD1(f::Formula,dat::AbstractDataFrame)
    mf = ModelFrame(f,dat)
    mm = ModelMatrix(mf)
    BolusSD1(mm.m[:,end],model_response(mf))
end
BolusSD1(ex::Expr,dat::AbstractDataFrame) = BolusSD1(Formula(ex),dat)

function BolusSD1f(pars::StridedVector,xv::StridedVector,tg::StridedVecOrMat)
    tg[2] = -xv[1]*(mm = pars[1]*(tg[1] = exp(-pars[2]*xv[1])))
    mm
end

pnames(m::BolusSD1) = ["V","K"]

function initpars{T<:FP}(m::BolusSD1{T})
    (n = size(m.x,2)) < 2 && return [one(T),exp(-one(T))]
    cc = hcat(ones(T,n),vec(m.x[1,:]))\log(m.y)
    [exp(cc[1]),-cc[2]]
end
