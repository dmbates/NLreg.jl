for (nm, mmf, nnl, nl) in ((:AsympReg, :AsympRegmmf, 1, 2),
                           (:AsympOff, :AsympOffmmf, 2, 1),
                           (:AsympOrig, :AsympOrigmmf, 1, 1),
                           (:Biexp, :Biexpmmf, 2, 2),
                           (:BolusSD1, :BolusSD1mmf, 1, 1),
                           (:Chwirut, :Chwirutmmf, 2, 1),
                           (:Gompertz, :Gompertzmmf, 2, 1),
                           (:Logis3P, :Logis3Pmmf, 2, 1),
                           (:Logis4P, :Logis4Pmmf, 2, 2),
                           (:MicMen, :MicMenmmf, 1, 1))
    @eval begin
        immutable $nm{T<:FP} <: PLregModF{T}
            x::Matrix{T}
            y::Vector{T}
            mu::Vector{T}
            resid::Vector{T}
            tgrad::Matrix{T}
            MMD::Array{T,3}
            mmf::Function
        end
        function $nm{T<:FP}(x::Vector{T},y::Vector{T})
            (n = length(x)) == length(y)|| throw(DimensionMismatch(""))
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
formula(::MicMen) = :(Vm*x/(K+x))
pnames(::MicMen) = [:Vm, :K, :x]
dim(::MicMen) = (1,1,1)

function MicMenmmf(nlp,x,tg,MMD)
    @inbounds begin
        x1 = x[1]
        invde = inv(nlp[1] + x1)
        MMD[1,1] = -(tg[1] =  x1*invde)*invde
    end
end

function initpars(m::MicMen)
    x = vec(m.x); y = m.y
    length(y) < 2 && return [one(eltype(y))]
    cc = linreg(map(inv,x), map(inv,y))
    [cc[2] / cc[1]]
end

### Asymptotic Regression model
formula(::AsympReg) = :(Asym + (Asym - R0)*exp(-rc*x))
pnames(::AsympReg) = [:Asym,:R0,:rc,:x]
dim(::AsympReg) = (2,1,1) # conditionally linear, nonlinear, covariates

function AsympRegmmf(nlp,x,tg,MMD)
    x1 = x[1]
    tg[1] = 1 - (tg[2] = eKx = exp(-nlp[1]*x1))
    MMD[2,1] = -(MMD[1,1] = x1 * eKx)
end

function initpars(m::AsympReg)
    y = m.y
    mn,mx = extrema(y)
    [abs(linreg(vec(m.x), log(y - mn + (mx-mn)/4.))[2])]
end

### Asymptotic Regression model constrained to pass through the origin
formula(::AsympOrig) = :(Asym*(1 - exp(-rc*x)))
pnames(::AsympOrig) = [:Asym,:rc,:x]
dim(::AsympOrig) = (1,1,1)

function AsympOrigmmf(nlp,x,tg,MMD)
    x1 = x[1] 
    tg[1] =  1 - (ex = exp(-nlp[1]*x1))
    MMD[1,1] = x1*ex
end

function initpars(m::AsympOrig)
    y = m.y
    mn,mx = extrema(y)
    A0 = mx + (mx-mn)/4.
    [abs(mean(log(1 - y/A0) ./ vec(m.x)))]
end

### Asymptotic Regression model expressed with an offset in x
formula(::AsympOff) = :(Asym*(1-exp(-rc*(x-c0))))
pnames(::AsympOff) = [:Asym,:c0,:rc,:x]
dim(::AsympOff) = (1,2,1)

function AsympOffmmf(nlp,x,tg,MMD)
    tg[1] = 1 - (eKx = exp(-(K = nlp[2])*(x1 = x[1] - nlp[1])))
    MMD[1,1] = -K * eKx
    MMD[2,1] = x1 * eKx
end

function initpars(m::AsympOff)
    cc = coef(fit(AsympReg(vec(m.x),m.y))) # fit the model with AsympReg pars
    rc = cc[3]; A = cc[1]
    [-log(A/(A-cc[2]))/rc, rc]
end

### Bolus single dose in measured compartment
formula(::BolusSD1) = :(V*exp(-K*x))
pnames(::BolusSD1) = [:V,:K,:x]
dim(::BolusSD1) = (1,1,1)

function BolusSD1mmf(nlp,x,tg,MMD)
    x1 = x[1]
    nKx1 = -nlp[1] * x1            # negative K*x[1] where nlp[1] is K
    MMD[1,1] = -x1 * (tg[1] = exp(nKx1))
end

function initpars{T<:FP}(m::BolusSD1{T})
    (n = length(m.y)) < 2 && return [-one(T)]
    cc = linreg(vec(m.x[1,:]),log(m.y))
    [cc[2] < 0. ? -cc[2] : -one(T)]
end

### 3-parameter Logistic
formula(::Logis3P) = :(Asym/(1 + exp(-(x-xmid)/scal)))
pnames(::Logis3P) = [:Asym,:xmid,:scal,:x]
dim(::Logis3P) = (1,2,1)

function Logis3Pmmf(nlp,x,tg,MMD)
    scal = nlp[2]
    nd = nlp[1] - x[1]                # negative difference from xmid
    ed = exp(nd/scal)                 # exp of standardized difference
    oped = 1 + ed
    tg[1] = 1/oped
    MMD[2,1] = -nd*(MMD[1,1] = -(ed/scal)/abs2(oped))/scal
end

function initpars(m::Logis3P)
    z = copy(m.y)
    if (minz = minimum(z)) < 0.       # ensure minimum(z) is positive
        z -= 1.05 * minz
    end
    z /= (1.05 * maximum(z))          # all z values in (0,1)
    linreg(log(z ./ (one(eltype(z)) - z)),vec(m.x))
end

### 4-parameter Logistic 
formula(::Logis4P) = :(A + (B-A)/(1 + exp(-(x-xmid)/scal)))
pnames(::Logis4P) = [:A,:B,:xmid,:scal,:x]                  
dim(::Logis3P) = (2,2,1)

function Logis4Pmmf(nlp,x,tg,MMD)
    scal = nlp[2]
    nd = nlp[1] - x[1]                # negative difference from xmid
    ed = exp(nd/scal)                 # exp of standardized difference
    oped = 1 + ed
    tg[1] = ed*(tg[2] = 1/oped)
    MMD[1,2] = -(bb = MMD[1,1] = ed/(scal * abs2(oped)))
    MMD[2,1] = -(MMD[2,2] = nd * bb/scal)
end

function initpars(m::Logis4P)
    y = m.y
    x = vec(m.x)
    rg = range(y)
    z = (y - minimum(y) + rg/20)/(1.1rg)
    linreg(log(z ./ (1 - z)),x)
end

### Biexponential
formula(::Biexp) = :(A1 * exp(-K1*x) + A2 * exp(-K2*x))
pnames(::Biexp) = [:A1,:A2,:K1,:K2,:x]
dim(::Biexp) = (2,2,1)
function Biexpmmf(nlp,x,tg,MMD)
    x1 = x[1]
    tg[1] = eK1 = exp(-nlp[1]*x1)
    tg[2] = eK2 = exp(-nlp[2]*x1)
    MMD[1,1] = -x1 * eK1
    MMD[2,2] = -x1 * eK2
end

function initpars(m::Biexp)
    x = copy(vec(m.x))
    perm = sortperm(x)
    x = x[perm]
    y = copy(m.y)[perm]
    n = length(y)
    nlast = max(3, n>>1)  
    ll = (n - nlast + 1):n      # "last half" of data but min of 3 els
    cc = linreg(x[ll],log(y[ll]))
    rc2 = abs(cc[2])
    [abs(linreg(x,log(abs(y - exp(cc[1] - rc2 * x))))[2]),rc2]
end

### Gompertz

formula(::Gompertz) = :(Asym*exp(-b2*b3^x))
pnames(::Gompertz) = [:Asym,:a,:b,:x]
dim(::Gompertz) = (1,2,1)
function Gompertzmmf(nlp,x,tg,MMD)
    x1 = x[1]
    a = nlp[1]
    b = nlp[2]
    p1 = b^x1
    tg[1] = e1 = exp(-a*p1)
    MMD[1,1] = -p1 * e1
    MMD[2,1] = -e1 * a * b^(x1 - 1) * x1
end

formula(::Chwirut) = :(R0*exp(-rc*x)/1+m*x)
pnames(::Chwirut) = [:R0,:rc,:m,:x]
dim(::Chwirut) = (1,2,1)
function Chwirutmmf(nlp,x,tg,MMD)
    x1 = x[1]
    et = exp(-nlp[1]*x1)
    denom = 1 + nlp[2] * x1
    tg[1] = et/denom
    etx = et * x1
    MMD[2,1] = (MMD[1,1] = -etx/denom)/denom
end


initpars(m::Chwirut) = [-(linreg(vec(m.x),log(m.y))[2]), mean(m.x)]

immutable Logsd1{T<:FP} <: NLregModF{T}
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
    [cc[2] < 0. ? [cc[1],log(-cc[2])] : [cc[1],-one(T)]]
end

pnames(m::Logsd1) = [:logV,:logK,:x]
