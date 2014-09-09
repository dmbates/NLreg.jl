## Specialized type of nonlinear mixed-effects model with a single
## grouping factor and random effects for each parameter in the
## nonlinear model

type SimpleNLMM{T<:FP} <: NLMM{T}
    m::NLregModF{T}
    inds::Vector               # grouping factor indices
    nrep::Vector               # run-length encoding of inds
    lambda::AbstractMatrix{T}  # template block for Lambda
    L::Vector{Matrix{T}}       # should this be a 3-D array?
    beta::Vector{T}            # fixed-effects parameter vector
    u::Matrix{T}               # spherical random-effects values
    delu::Matrix{T}            # increment in the PNLS algorithm
    b::Matrix{T}               # random effects on original scale
    phi::Matrix{T}             # phi := beta + lambda * (u + fac*delu)
    nAGQ::Int                  # adaptive Gauss-Hermite quadrature points
    mxpnls::Int                # maximum number of PNLS iterations
    minfac::T                  # minimum step factor in PNLS
    tolsqr::T                  # squared tolerance for orthogonality conv.crit.
end
function SimpleNLMM{T<:FP}(m::NLregModF{T},inds::Vector,
                           lambda::AbstractMatrix{T},beta::Vector{T})
    n = size(m)[end]; p = npars(m); np = n*p; ui, nrep = rle(inds); ni = length(ui)
    length(inds) == n || error("length(inds) = $(length(inds)), should be $n")
    isperm(ui) ||
        error("similar inds must be adjacent and unique(inds) a permutation")
    size(lambda) == (p,p) ||
        error("size(lambda) = $(size(lambda)) should be $((p,p))")
    L = Matrix{T}[eye(T,p) for i in 1:ni]
    u = zeros(T,(p,ni))
    SimpleNLMM(m,inds,nrep,lambda,L,beta,u,copy(u),similar(u),similar(u),
               1,300,convert(T,0.5^9),convert(T,1e-8))
end
function SimpleNLMM(nl::Union(PLinearLS,NonlinearLS),inds::Vector,ltype::DataType)
    lambda = ltype(eye(length(coef(nl))))
    if ltype == Triangular
        lambda.uplo = 'L'
        lambda.unitdiag = 'N'
    end
    SimpleNLMM(deepcopy(nl.m),inds,lambda,coef(nl))
end

StatsBase.coef(nm::SimpleNLMM) = copy(nm.beta)

nlregmod(nm::SimpleNLMM) = nm.m

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

pwrss(nm::SimpleNLMM) = sumsq(nm.m.resid) + sumsq(vec(nm.u))

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
        BLAS.gemv!('N',1.,g,sub(rr,ii),-1.,du)
        ## Don't need to check for singularity in potrf! result b/c adding I
        ## TODO: use two applications of trsv and evaluate the numerator of the conv. crit.
        LAPACK.potrs!('L',LAPACK.potrf!('L',BLAS.syrk!('L','N',1.,g,1.,copy!(L[i],ee)))[1],du)
    end
    prss!(nm,one(T))                    # evaluate prss with full increment
end

## increment the spherical random effects as nm.u += nm.delu * fac and zero out nm.delu
function incr!{T<:FP}(nm::SimpleNLMM{T}, fac::T)
    fma!(nm.u,nm.delu,fac)
    fill!(nm.delu,zero(T))
end

## evaluate the approximate deviance using adaptive Gauss-Hermite quadrature
function StatsBase.deviance{T<:FP}(nm::SimpleNLMM{T})
    nm.nAGQ == 1 || error("General adaptive Gauss-Hermite quadrature not yet written")
    n = nobs(nm)
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

function Base.show(io::IO, m::SimpleNLMM)
    mm = nlregmod(m)
    mstr = m.nAGQ == 1 ? "Laplace approximation" : "adaptive Gauss-Hermite quadrature ($(m.nAGQ))"
    println(io, "Simple, nonlinear mixed-effects model fit by ", mstr)
    oo = deviance(m)
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
    @printf(io," Number of obs: %d; levels of grouping factor: %d", nobs(m), size(m.u,2))
    println(io); println(io)

    println(io, "Fixed effects parameters:")
    show(io, coeftable(m))
    println(io)
end

Base.std(nm::SimpleNLMM) = scale(nm) * [diag(nm.lambda),1.]
