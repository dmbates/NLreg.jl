abstract NLMM <: StatisticalModel

type SimpleNLMM{T<:FP} <: NLMM
    m::NLregMod{T}
    inds::Vector
    nrep::Vector
    lambda::AbstractMatrix{T}
    L::Vector{Matrix{T}}  # should this be a 3-D array?
    beta::Vector{T}       # fixed-effects parameter vector
    u::Matrix{T}          # spherical random-effects values
    delu::Matrix{T}       # increment in the PNLS algorithm
    b::Matrix{T}          # random effects on original scale
    phi::Matrix{T}        # phi := beta + lambda * (u + fac*delu)
    mxpnls::Int           # maximum number of PNLS iterations
    minfac::T
    tolsqr::T
end
function SimpleNLMM{T<:FP}(m::NLregMod{T},inds::Vector,lambda::AbstractMatrix{T},beta::Vector{T})
    p,n = size(m); np = n*p; ui = unique(inds); ni = length(ui)
    isperm(ui) || error("unique(inds) should be a permutation")
    length(inds) == n || error("length(inds) = $(length(inds)), should be $n")
    nrep = rle(inds)[2]; length(nrep) == ni || error("similar inds must be adjacent")
    size(lambda) == (p,p) || error("size(lambda) = $(size(lambda)) should be $((p,p))")
    L = Matrix{T}[eye(T,p) for i in 1:ni]
    u = zeros(T,(p,ni))
    SimpleNLMM(m,inds,nrep,lambda,L,beta,u,copy(u),similar(u),similar(u),30,convert(T,0.5^9),convert(T,1e-8))
end
## ToDo Add an external constructor using Formula/Data

## Multiply u by lambda in place creating the b values
u2b!{T<:FP}(lambda::Diagonal{T},u::Matrix{T}) = scale!(lambda.diag,u)
u2b!{T<:FP}(lambda::Triangular{T},u::Matrix{T}) = trmm!('L','L','N','N',1.,lambda,u)

## penalized residual sum of squares
function prss!{T<:FP}(nm::SimpleNLMM{T},fac::T)
    b = nm.b                  # random effects on original scale
    copy!(b,nm.u)             # initialize with u
    fma!(b,nm.delu,fac)       # add fac*delu
    ssu = sqsum(b)            # record squared length of u + fac(delu)
    u2b!(nm.lambda,b)         # convert to b scale
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
    mm = u2b!(nm.lambda,copy(m.tgrad)) # multiply transposed gradient by lambda
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

## Determine the conditional mode of the random effects using penalized nonlinear least squares
function pnlsfit!{T<:FP}(nm::SimpleNLMM{T})
    m = nm.m; id = nm.inds; RX = nm.RX; lam = nm.lambda; L = nm.L; u = nm.u; delu = nm.delu
    oldprss = prss!(nm,0.); conv = one(T)
    for i in 1:nm.mxpnlsit
        newprss = updtL!(nm); f = one(T)
        while oldprss < newprss && f >= nm.minfac
            f /= convert(T,2)
            newprss = prss!(nm,f)
        end
        f < nm.minfac && error("Failure to reduce prss at u = ",nm.u)
        incr!(pp,f)
        (conv = (oldprss - newprss)/newprss) < nm.tolsqr && break
    end
    conv >= nm.tolsqr && error("Failure to converge in ", nm.mxpnlsit, " iterations")
end
