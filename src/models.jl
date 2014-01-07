immutable MicMen{T<:FP} <: PLregMod{T}
    x::Matrix{T}
    y::Vector{T}
    mu::Vector{T}
    resid::Vector{T}
    tgrad::Matrix{T}
    MMD::Array{T,3}
    mmf::Function
end
function MicMenmmf(nlpars::StridedVector,x::StridedVector,
                   tg::StridedVector,MMD::StridedMatrix)
    x1 = x[1]
    denom = nlpars[1] + x1
    MMD[1,1] = -(tg[1] =  x1/denom)/denom
end
function MicMen{T<:FP}(x::Vector{T},y::Vector{T})
    n = length(x); length(y) == n || error("Dimension mismatch")
    MicMen(reshape(x,(1,n)),y,similar(x),similar(x),Array(T,(2,n)),
           zeros(T,(1,1,n)),MicMenmmf)
end
MicMen(x::DataVector,y::DataVector) = MicMen(float(x),float(y))
function MicMen(f::Formula,dat::AbstractDataFrame)
    mf = ModelFrame(f,dat)
    mat = ModelMatrix(mf).m
    rr = model_response(mf)
    T = promote_type(eltype(mat),eltype(rr))
    MicMen(convert(Vector{T},mat[:,end]),convert(Vector{T},rr))
end
MicMen(ex::Expr,dat::AbstractDataFrame) = MicMen(Formula(ex),dat)

function initpars(m::MicMen)
    y = m.y; T = eltype(y)
    (n = length(y)) < 2 && return [m.y[1],one(T)]
    cc = hcat(ones(T,n),1. ./ vec(m.x[1,:]))\ (1. ./ m.y)
    [1.,cc[2]] ./ cc[1]
end
    
pnames(m::MicMen) = ["Vm", "K"]

immutable AsympReg{T<:FP} <: PLregMod{T}
    x::Matrix{T}
    y::Vector{T}
    mu::Vector{T}
    resid::Vector{T}
    tgrad::Matrix{T}
    MMD::Array{T,3}
    mmf::Function
end
function AsympRegmmf(nlpars::StridedVector,x::StridedVector,
                    tg::StridedVector,MMD::StridedMatrix)
    x1 = x[1];
    MMD[2,1] = -x1 *(tg[2] = exp(-nlpars[1]*x1))
end
function AsympReg{T<:FP}(x::Vector{T},y::Vector{T})
    n = length(x); length(y) == n || error("Dimension mismatch")
    AsympReg(reshape(x,(1,n)),y,similar(x),similar(x),ones(T,(3,n)),
           zeros(T,(1,2,n)),AsympRegmmf)
end
AsympReg(x::DataVector,y::DataVector) = AsympReg(float(x),float(y))
function AsympReg(f::Formula,dat::AbstractDataFrame)
    mf = ModelFrame(f,dat)
    mat = ModelMatrix(mf).m
    rr = model_response(mf)
    T = promote_type(eltype(mat),eltype(rr))
    AsympReg(convert(Vector{T},mat[:,end]),convert(Vector{T},rr))
end
AsympReg(ex::Expr,dat::AbstractDataFrame) = AsympReg(Formula(ex),dat)
pnames(m::AsympReg) = ["Asym","R0","rc"]

function initpars(m::AsympReg)
    y = m.y; T = eltype(y); n = length(y)
    miny = minimum(y); maxy = maximum(y)
    p3 = abs(linreg(m.x[1,:]', log(y .- (miny - convert(T,0.25)*(maxy-miny))))[2])
    updtMM!(m,[p3])
    [sub(m.tgrad,1:2,:)'\m.y, p3]
end

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

function Logsd1f(p::StridedVector,x::StridedVector,tg::StridedVector)
    x1 = x[1]; V = exp(p[1]); K = exp(p[2])
    tg[2] = -x1*K*(mm = V*exp(-K*x1))
    tg[1] = V*mm
    mm
end

function initpars{T<:FP}(m::Logsd1{T})
    (n = length(m.y)) < 2 && return [zero(T),-one(T)]
    cc = hcat(ones(n),vec(m.x[1,:]))\log(m.y)
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



