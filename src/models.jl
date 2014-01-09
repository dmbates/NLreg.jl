
function Logis3Pmmf(nlpars::StridedVector,x::StridedVector,
                    tg::StridedVector,MMD::StridedMatrix)
    scal = nlpars[2]
    ed = exp((nlpars[1] - x[1])/scal) # standardized difference from xmid
    T = typeof(ed); oo = one(T)
    oped = oo + ed
    tg[1] = oo/oped
    MMD[2,1] = -(MMD[1,1] = -ed/scal/abs2(oped)) * ed
end

for (nm, mmf, nnl, nl) in ((:MicMen, :MicMenmmf, 1, 1),
                           (:AsympReg, :AsympRegmmf, 1, 2),
                           (:AsympOrig, :AsympOrigmmf, 1, 1),
                           (:LogBolusSD1, :LogBolusSD1mmf, 1, 1),
                           (:Logis3P, :Logis3Pmmf, 2, 1),
                           (:Chwirut, :Chwirutmmf, 2, 1))
    @eval begin
        immutable $nm{T<:FP} <: PLregMod{T}
            x::Matrix{T}
            y::Vector{T}
            mu::Vector{T}
            resid::Vector{T}
            tgrad::Matrix{T}
            MMD::Array{T,3}
            mmf::Function
        end
        function $nm{T<:FP}(x::Vector{T},y::Vector{T})
            n = length(x); length(y) == n || error("Dimension mismatch")
            $nm(reshape(x,(1,n)),y,similar(y),similar(y),ones(T,($(nl + nnl),n)),
                zeros(T,($nnl,$nl,n)),$mmf)
        end
        $nm(x::DataVector,y::DataVector) = $nm(float(x),float(y))
        function $nm(f::Formula,dat::AbstractDataFrame)
            mf = ModelFrame(f,dat)
            mat = ModelMatrix(mf).m
            rr = model_response(mf)
            T = promote_type(eltype(mat),eltype(rr))
            $nm(convert(Vector{T},mat[:,end]),convert(Vector{T},rr))
        end
        $nm(ex::Expr,dat::AbstractDataFrame) = $nm(Formula(ex),dat)
    end
end
    
### Michaelis-Menten model for enzyme kinetics
function MicMenmmf(nlpars::StridedVector,x::StridedVector,
                   tg::StridedVector,MMD::StridedMatrix)
    x1 = x[1]
    denom = nlpars[1] + x1
    MMD[1,1] = -(tg[1] =  x1/denom)/denom
end

pnames(m::MicMen) = ["Vm", "K"]

function initpars{T<:FP}(m::MicMen{T})
    y = m.y
    oo = one(T)
    (n = length(y)) < 2 && return [m.y[1],oo]
    cc = linreg(oo ./ vec(m.x[1,:]), (oo ./ m.y))
    [oo,cc[2]] ./ cc[1]
end

### Asymptotic Regression model
function AsympRegmmf(nlpars::StridedVector,x::StridedVector,
                    tg::StridedVector,MMD::StridedMatrix)
    x1 = x[1];
    MMD[2,1] = -x1 *(tg[2] = exp(-nlpars[1]*x1))
end

pnames(m::AsympReg) = ["Asym","delR0","rc"]

function initpars{T<:FP}(m::AsympReg{T})
    y = m.y
    n = length(y)
    p3 = abs(linreg(vec(m.x[1,:]), log(y .- (minimum(y) - range(y)/4)))[2])
    [updtMM!(m,[p3])'\y, p3]
end

### Asymptotic Regression model constrained to pass through the origin
function AsympOrigmmf(nlpars::StridedVector,x::StridedVector,
                      tg::StridedVector,MMD::StridedMatrix)
    x1 = x[1]; ex = exp(-nlpars[1]*x1)
    MMD[1,1] = x1*ex
    tg[1] =  one(typeof(x1)) - ex
end

pnames(m::AsympOrig) = ["V","ke"]

function initpars{T<:FP}(m::AsympOrig{T})
    y = m.y; A0 = maximum(y) + range(y)/4
    rc =  abs(mean(log(one(T) - y/A0) ./ vec(m.x)))
    [vec(updtMM!(m,[rc]))\y, rc]
end

### Bolus single dose in measured compartment using logK
function LogBolusSD1mmf(nlpars::StridedVector,x::StridedVector,
                        tg::StridedVector,MMD::StridedMatrix)
    nKx1 = -exp(nlpars[1]) * x[1] 
    MMD[1,1] = nKx1 * (tg[1] = exp(nKx1))
end

pnames(m::LogBolusSD1) = ["V","lK"]

function initpars{T<:FP}(m::LogBolusSD1{T})
    (n = length(m.y)) < 2 && return [zero(T),-one(T)]
    cc = hcat(ones(n),vec(m.x[1,:]))\log(m.y)
    cc[2] < 0. ? [exp(cc[1]),log(-cc[2])] : [exp(cc[1]),-one(T)]
end

### 3-parameter Logistic 
function Logis3Pmmf(nlpars::StridedVector,x::StridedVector,
                    tg::StridedVector,MMD::StridedMatrix)
    scal = nlpars[2]
    nd = nlpars[1] - x[1]        # negative difference from xmid
    ed = exp(nd/scal)            # exp of standardized difference
    oo = one(typeof(ed))
    oped = oo + ed
    tg[1] = oo/oped
    MMD[2,1] = -nd*(MMD[1,1] = -(ed/scal)/abs2(oped))/scal
end

pnames(m::Logis3P) = ["Asym","xmid","scal"]

function initpars{T<:FP}(m::Logis3P{T})
    z = m.y
    if (minz = minimum(z)) < zero(T)  # ensure minimum(z) is positive
        z -= 1.05 * minz
    end
    z = z/(1.05 * maximum(z))           # all z values in (0,1)
    cc = linreg(log(z ./ (one(T) - z)),vec(m.x))
    [updtMM!(m,cc)'\m.y,cc]
end

function Chwirutmmf(nlpars::StridedVector,x::StridedVector,
                    tg::StridedVector,MMD::StridedMatrix)
    x1 = x[1]
    et = exp(-nlpars[1]*x1)
    denom = 1 + nlpars[2] * x1
    tg[1] = et/denom
    etx = et * x1
    MMD[2,1] = (MMD[1,1] = -etx/denom)/denom
end

pnames(m::Chwirut) = ["R0","rc","m"]

function initpars(m::Chwirut)
    npars = [-(linreg(vec(m.x),log(m.y))[2]), mean(m.x)]
    [vec(updtMM!(m,npars))\m.y,npars]
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
