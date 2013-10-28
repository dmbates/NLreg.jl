abstract NLMM{T<:FP} <: StatisticalModel

type SimpleNLMM{T<:FP} <: NLMM{T}
    m::NLregMod{T}
    inds::Vector               # grouping factor indices
    nrep::Vector               # run-length encoding of inds
    lambda::AbstractMatrix{T}  # template block for Lambda
    L::Vector{Matrix{T}}       # should this be a 3-D array?
    beta::Vector{T}            # fixed-effects parameter vector
    u::Matrix{T}               # spherical random-effects values
    delu::Matrix{T}            # increment in the PNLS algorithm
    b::Matrix{T}               # random effects on original scale
    phi::Matrix{T}             # phi := beta + lambda * (u + fac*delu)
    nAGQ::Int               # adaptive Gauss-Hermite quadrature points
    mxpnls::Int             # maximum number of PNLS iterations
    minfac::T               # minimum step factor in PNLS
    tolsqr::T               # squared tolerance for orthogonality conv.crit.
end
function SimpleNLMM{T<:FP}(m::NLregMod{T},inds::Vector,
                           lambda::AbstractMatrix{T},beta::Vector{T})
    p,n = size(m); np = n*p; ui, nrep = rle(inds); ni = length(ui)
    length(inds) == n || error("length(inds) = $(length(inds)), should be $n")
    isperm(ui) ||
        error("similar inds must be adjacent and unique(inds) a permutation")
    size(lambda) == (p,p) ||
        error("size(lambda) = $(size(lambda)) should be $((p,p))")
    L = Matrix{T}[eye(T,p) for i in 1:ni]
    u = zeros(T,(p,ni))
    SimpleNLMM(m,inds,nrep,lambda,L,beta,u,copy(u),similar(u),similar(u),
               1,30,convert(T,0.5^9),convert(T,1e-8))
end
function SimpleNLMM(nl::NonlinearLS,inds::Vector,lambda::AbstractMatrix)
    SimpleNLMM(deepcopy(nl.m),inds,lambda,coef(nl))
end

A_mul_B!(A::Diagonal,B::Matrix) = scale!(A.diag,B)
At_mul_B!(A::Diagonal,B::Matrix)= scale!(A.diag,B)
Ac_mul_B!(A::Diagonal,B::Matrix)= scale!(A.diag,B)
A_mul_B!(A::Triangular,B::Matrix) = trmm!('L',A.uplo,'N',A.unitdiag,one(eltype(A)),A.UL,B)
Ac_mul_B!(A::Triangular,B::Matrix) = trmm!('L',A.uplo,'T',A.unitdiag,one(eltype(A)),A.UL,B)

## penalized residual sum of squares
function prss!{T<:FP}(nm::SimpleNLMM{T},fac::T)
    b = nm.b                  # random effects on original scale
    copy!(b,nm.u)             # initialize with u
    fma!(b,nm.delu,fac)       # add fac*delu
    ssu = sumsq(b)            # record squared length of u + fac(delu)
    A_mul_B!(nm.lambda,b)     # convert to b scale
    updtmu!(nm.m,broadcast!(+,nm.phi,b,nm.beta),nm.inds) + ssu # rss + penalty
end
prss!{T<:FP}(nm::SimpleNLMM{T}) = prss!(nm,zero(T))

residuals(nm::SimpleNLMM) = nm.m.resid

## logdet of L, the Cholesky factor
function ldL2{T<:FP}(m::Matrix{T})
    dd = zero(T)
    for i in 1:size(m,1) dd += log(m[i,i]) end
    dd + dd
end
ldL2{T<:FP}(nm::SimpleNLMM{T}) = sum(ldL2,nm.L)

## Update the L array using the current values of tgrad
function updtL!{T<:FP}(nm::SimpleNLMM{T})
    m = nm.m; u = nm.u; delu = copy!(nm.delu,nm.u); L = nm.L; ee = eye(T,size(L[1],1))
    mm = nm.lambda' * m.tgrad
    rr = residuals(m); nn = nm.nrep; offset = 0
    for i in 1:length(L)
        ni = nn[i]; ii = offset+(1:nn[i]); offset += ni;
        g = sub(mm,:,ii); du = sub(delu,:,i)
        gemv!('N',1.,g,sub(rr,ii),-1.,du)
        ## Don't need to check for singularity in potrf! result b/c adding I
        ## TODO: use two anmlications of trsv and evaluate the numerator of the conv. crit.
        potrs!('L',potrf!('L',syrk!('L','N',1.,g,1.,copy!(L[i],ee)))[1],du)
    end
    prss!(nm,one(T))                    # evaluate prss with full increment
end

## increment the spherical random effects as nm.u += nm.delu * fac and zero out nm.delu
function incr!{T<:FP}(nm::SimpleNLMM{T}, fac::T)
    fma!(nm.u,nm.delu,fac)
    fill!(nm.delu,zero(T))
end

## evaluate the approximate deviance using adaptive Gauss-Hermite quadrature
function deviance{T<:FP}(nm::SimpleNLMM{T})
    nm.nAGQ == 1 || error("General adaptive Gauss-Hermite quadrature not yet written")
    p,n = size(nm.m)
    ldL2(nm) + n * (one(T) + log(2pi*prss!(nm)/n))
end

## Determine the conditional mode of the random effects using penalized nonlinear least squares
function pnls!{T<:FP}(nm::SimpleNLMM{T},u0::Matrix{T};verbose=false)
    copy!(nm.u,u0); fill!(nm.delu,zero(T)) # start from a fixed position
    oldprss = prss!(nm,0.); conv = one(T);
    for i in 1:nm.mxpnls
        newprss = updtL!(nm); f = one(T)
        verbose && @printf(" %12f, %12f\n", oldprss, newprss)
        while oldprss < newprss && f >= nm.minfac
            f /= convert(T,2)
            newprss = prss!(nm,f)
            verbose && @printf("  f = %8e: %12f\n", f, newprss)
        end
        f < nm.minfac && error("Failure to reduce prss at u = ",nm.u)
        incr!(nm,f)
        (conv = (oldprss - newprss)/newprss) < nm.tolsqr && break
        oldprss = newprss
    end
    conv >= nm.tolsqr && error("Failure to converge in ", nm.mxpnls, " iterations")
    deviance(nm)
end
pnls!{T<:FP}(nm::SimpleNLMM{T};verbose=true) = pnls!(nm, zeros(T,size(nm.u));verbose=verbose)

theta(nm::NLMM) = theta(nm.lambda)
theta(lambda::Diagonal) = lambda.diag
function theta(m::Triangular)
    n = size(m,1); v = Array(eltype(m),n*(n+1)>>1); pos = 0; UL = m.UL
    for j in 1:n, i in (m.uplo == 'L' ? (j:n) : (1:j)) v[pos += 1] = UL[i,j] end
    v
end

theta!{T}(lambda::Diagonal{T},th::Vector{T}) = copy!(lambda.diag,th)
function theta!{T}(m::Triangular{T},th::Vector{T})
    n = size(m,1); pos = 0; UL = m.UL
    length(th) == n*(n+1)>>1 || error("length(th) = $(length(th)), should be $(n*(n+1)/2)")
    for j in 1:n, i in (m.uplo == 'L' ? (j:n) : (1:j)) UL[i,j] = th[pos += 1] end
    m
end

lowerbd{T<:FP}(nm::NLMM{T}) = [fill(-inf(T),length(nm.beta)),lowerbd(nm.lambda)]
lowerbd{T}(lambda::Diagonal{T}) = zeros(T,size(lambda,1))
function lowerbd{T}(m::Triangular{T})
    n = size(m,1); v = fill(-inf(T),n*(n+1)>>1); pos = 1; zt = zero(T)
    for j in 1:n, i in (m.uplo == 'L' ? (j:n) : (1:j))
        i == j && (v[pos] = zt)
        pos += 1
    end
    v
end

function setpars!{T<:FP}(nm::NLMM{T},pars::Vector{T})
    bb = nm.beta; p = length(bb); copy!(bb, sub(pars,1:p))
    theta!(nm.lambda,pars[(p+1):end])
    nm
end
    
function fit(nm::NLMM; verbose=false)
    pnls!(nm; verbose=verbose)
    u0 = copy(nm.u)
    th = theta(nm); nth = length(th)
    pars = [nm.beta,th]
    o = Opt(:LN_BOBYQA,length(pars))
    ftol_abs!(o, 1e-6)    # criterion on deviance changes
    xtol_abs!(o, 1e-6)    # criterion on all parameter value changes
    lower_bounds!(o, lowerbd(nm))
    function obj(x::Vector{Float64}, g::Vector{Float64})
        length(g) == 0 || error("gradient evaluations are not provided")
        pnls!(setpars!(nm,x),u0;verbose=verbose)
    end
    if verbose
        count = 0
        function vobj(x::Vector{Float64}, g::Vector{Float64})
            count += 1
            length(g) == 0 || error("gradient evaluations are not provided")
            val = pnls!(setpars!(nm,x),u0)
            print("f_$count: $(round(val,5)), [")
            showcompact(x[1])
            for i in 2:length(x) print(","); showcompact(x[i]) end
            println("]")
            val
        end
        min_objective!(o, vobj)
    else
        min_objective!(o, obj)
    end
    fmin, xmin, ret = optimize!(o, pars)
    verbose && println(ret)
    setpars!(nm,xmin)
    nm
end

pwrss(nm::NLMM) = sumsq(nm.m.resid) + sumsq(vec(nm.u))

function scale(nm::NLMM, sqr=false)
    m = nm.m; p,n = size(m); ssqr = pwrss(nm)/n
    sqr ? ssqr : sqrt(ssqr)
end

std(nm::SimpleNLMM) = scale(nm) * [diag(nm.lambda),1.]

function show(io::IO, m::SimpleNLMM)
    fit(m); mm = m.m; p,n = size(mm)
    mstr = m.nAGQ == 1 ? "Laplace approximation" : "adaptive Gauss-Hermite quadrature ($(m.nAGQ))"
    @printf(io, "Simple, nonlinear mixed-effects model fit by %s\n", mstr)
    oo = deviance(pp)
    @printf(io, " logLik: %f, deviance: %f", -oo/2., oo)
    println(io); println(io)

    
    @printf(io, " Variance components:\n                Variance    Std.Dev.\n")
    stdm = std(m); fnms = vcat(pnames(mm),"Residual")
    for i in 1:length(fnms)
        si = stdm[i]
        print(io, " ", rpad(fnms[i],12))
        @printf(io, " %10f  %10f\n", abs2(si[1]), si[1])
        for j in 2:length(si)
            @printf(io, "             %10f  %10f\n", abs2(si[j]), si[j])
        end
    end
    @printf(io," Number of obs: %d; levels of grouping factors: %d", n, size(pp.u,2))
    println(io)
    @printf(io,"\n  Fixed-effects parameters:\n")
    tstrings = split(string(coeftable(m)),'\n')
    for i in 2:p+2 print(io,tstrings[i]); print(io,"\n") end
end
