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
pnames(m::AsympReg) = ["Asym","delR0","rc"]

function initpars(m::AsympReg)
    y = m.y; T = eltype(y); n = length(y)
    miny = minimum(y); maxy = maximum(y)
    p3 = abs(linreg(m.x[1,:]', log(y .- (miny - convert(T,0.25)*(maxy-miny))))[2])
    updtMM!(m,[p3])
    [sub(m.tgrad,1:2,:)'\m.y, p3]
end

immutable AsympOrig{T<:FP} <: PLregMod{T}
    x::Matrix{T}
    y::Vector{T}
    mu::Vector{T}
    resid::Vector{T}
    tgrad::Matrix{T}
    MMD::Array{T,3}
    mmf::Function
end
function AsympOrigmmf(nlpars::StridedVector,x::StridedVector,
                      tg::StridedVector,MMD::StridedMatrix)
    x1 = x[1]; ex = exp(-nlpars[1]*x1)
    MMD[1,1] = x1*ex
    tg[1] =  one(typeof(x1)) - ex
end
function AsympOrig{T<:FP}(x::Vector{T},y::Vector{T})
    n = length(x); length(y) == n || error("Dimension mismatch")
    AsympOrig(reshape(x,(1,n)),y,similar(x),similar(x),Array(T,(2,n)),
              zeros(T,(1,1,n)),AsympOrigmmf)
end
AsympOrig(x::DataVector,y::DataVector) = AsympOrig(float(x),float(y))
function AsympOrig(f::Formula,dat::AbstractDataFrame)
    mf = ModelFrame(f,dat)
    mat = ModelMatrix(mf).m
    rr = model_response(mf)
    T = promote_type(eltype(mat),eltype(rr))
    AsympOrig(convert(Vector{T},mat[:,end]),convert(Vector{T},rr))
end
AsympOrig(ex::Expr,dat::AbstractDataFrame) = AsympOrig(Formula(ex),dat)
pnames(m::AsympOrig) = ["V","ke"]
function initpars{T<:FP}(m::AsympOrig{T})
    y = m.y; A0 = maximum(y) + range(y)/4
    rc =  abs(mean(log(one(T) - y/A0) ./ vec(m.x)))
    [vec(updtMM!(m,[rc]))\y, rc]
end

immutable Logsd1{T<:FP} <: NLregMod{T}
    x::Matrix{T}
    y::Vector{T}
    mu::Vector{T}
    resid::Vector{T}
    tgrad::Matrix{T}
    f::Function
end
function Logsd1f(p::StridedVector,x::StridedVector,tg::StridedVector)
    V = exp(p[1]); nKx1 = -exp(p[2])*x[1] # negative K * x[1]
    tg[2] = nKx1 * (mm = V*exp(nKx1))
    tg[1] = V*mm
    mm
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

function initpars{T<:FP}(m::Logsd1{T})
    (n = length(m.y)) < 2 && return [zero(T),-one(T)]
    cc = hcat(ones(n),vec(m.x[1,:]))\log(m.y)
    cc[2] < 0. ? [cc[1],log(-cc[2])] : [cc[1],-one(T)]
end

pnames(m::Logsd1) = ["logV","logK"]

immutable LogBolusSD1{T<:FP} <: PLregMod{T}
    x::Matrix{T}
    y::Vector{T}
    mu::Vector{T}
    resid::Vector{T}
    tgrad::Matrix{T}
    MMD::Array{T,3}
    mmf::Function
end
function LogBolusSD1mmf(nlpars::StridedVector,x::StridedVector,
                        tg::StridedVector,MMD::StridedMatrix)
    nKx1 = -exp(nlpars[1]) * x[1] 
    MMD[1,1] = nKx1 * (tg[1] = exp(nKx1))
end
function LogBolusSD1{T<:FP}(x::Vector{T},y::Vector{T})
    n = length(x); length(y) == n || error("Dimension mismatch")
    LogBolusSD1(reshape(x,(1,n)),y,similar(x),similar(x),Array(T,(2,n)),
                zeros(T,(1,1,n)),LogBolusSD1mmf)
end
LogBolusSD1(x::DataVector,y::DataVector) = LogBolusSD1(float(t),float(y))
function LogBolusSD1(f::Formula,dat::AbstractDataFrame)
    mf = ModelFrame(f,dat)
    x = ModelMatrix(mf).m[:,end]
    y = model_response(mf)
    T = promote_type(eltype(x),eltype(y))
    LogBolusSD1(convert(Vector{T},x),convert(Vector{T},y))
end
LogBolusSD1(ex::Expr,dat::AbstractDataFrame) = LogBolusSD1(Formula(ex),dat)

pnames(m::LogBolusSD1) = ["V","lK"]

function initpars{T<:FP}(m::LogBolusSD1{T})
    (n = length(m.y)) < 2 && return [zero(T),-one(T)]
    cc = hcat(ones(n),vec(m.x[1,:]))\log(m.y)
    cc[2] < 0. ? [exp(cc[1]),log(-cc[2])] : [exp(cc[1]),-one(T)]
end

immutable Logis3P{T<:FP} <: PLregMod{T} # three parameter logistic
    x::Matrix{T}
    y::Vector{T}
    mu::Vector{T}
    resid::Vector{T}
    tgrad::Matrix{T}
    MMD::Array{T,3}
    mmf::Function
end
function Logis3Pmmf(nlpars::StridedVector,x::StridedVector,
                    tg::StridedVector,MMD::StridedMatrix)
    scal = nlpars[2]
    ed = exp((nlpars[1] - x[1])/scal) # standardized difference from xmid
    T = typeof(ed); oo = one(T)
    oped = oo + ed
    tg[1] = oo/oped
    MMD[2,1] = -(MMD[1,1] = -ed/scal/abs2(oped)) * ed
end
function Logis3P{T<:FP}(x::Vector{T},y::Vector{T})
    n = length(x); length(y) == n || error("Dimension mismatch")
    Logis3P(reshape(x,(1,n)),y,similar(x),similar(x),Array(T,(3,n)),
            zeros(T,(2,1,n)),Logis3Pmmf)
end

pnames(m::Logis3P) = ["Asym","xmid","scal"]
