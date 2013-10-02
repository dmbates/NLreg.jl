type SimplePopPK{T<:FP} <: StatisticalModel
    m::NLregMod{T}
    inds::Vector
    lambda::AbstractMatrix{T}
    L::Vector{Matrix{T}}  # should this be a 3-D array?
    beta::Vector{T}       # fixed-effects parameter vector
    u::Matrix{T}          # spherical random-effects values
    delu::Matrix{T}       # increment in the PNLS algorithm
    b::Matrix{T}
    phi::Matrix{T}      # parameter matrix - phi := beta + lambda * (u + fac*delu)
end
function SimplePopPK{T<:FP}(m::NLregMod{T},inds::Vector,lambda::AbstractMatrix{T},beta::Vector{T})
    p,n = size(m); np = n*p; ui = unique(inds); ni = length(ui)
    isperm(ui) || error("unique(inds) should be a permutation")
    length(inds) == n || error("length(inds) = $(length(inds)) should be $n")
    size(lambda) == (p,p) || error("size(lambda) = $(size(lambda)) should be $((p,p))")
    L = Matrix{T}[eye(T,p) for i in 1:ni]
    u = zeros(T,(p,ni))
    SimplePopPK(m,inds,lambda,L,beta,u,copy(u),similar(u),similar(u))
end
## ToDo Add an external constructor using Formula/Data

## Multiply u by lambda in place creating the b values
u2b!{T<:FP}(lambda::Diagonal{T},u::Matrix{T}) = scale!(lambda.diag,u)
u2b!{T<:FP}(lambda::Triangular{T},u::Matrix{T}) = trmm!('L','L','N','N',1.,lambda,u)

## penalized residual sum of squares
function prss!{T<:FP}(pp::SimplePopPK{T},fac::T)
    b = pp.b                  # random effects on original scale
    copy!(b,pp.u)             # initialize with u
    fma!(b,pp.delu,fac)       # add fac*delu
    ssu = sqsum(b)            # record squared length of u + fac(delu)
    u2b!(pp.lambda,b)         # convert to b scale
    updtmu!(pp.m,broadcast!(+,pp.phi,b,pp.beta),pp.inds) + ssu # rss + penalty
end

residuals(pp::SimplePopPK) = pp.m.resid

## logdet of L, the Cholesky factor
function ldL2{T<:FP}(m::Matrix{T})
    dd = zero(T)
    for i in 1:size(m,1) dd += log(m[i,i]) end
    dd + dd
end
function ldL2{T<:FP}(pp::SimplePopPK{T})
    dd = zero(T); L = pp.L
    for i in 1:length(L) dd += ldL2(L[i]) end
    dd
end

function updtL!{T<:FP}(pp::SimplePopPK{T})
    m = pp.m; u = pp.u; L = pp.L; ee = eye(T,size(L[1],1))
    mm = u2b!(pp.lambda,copy(m.tgrad)) # multiply transposed gradient by lambda
    rr = residuals(m); nn = rle(pp.inds)[2]; offset = 0
    for i in 1:length(L)
        ni = nn[i]; ii = offset+(1:nn[i]); offset += ni; g = sub(mm,:,ii); ui = sub(u,:,i)
        gemv!('N',1.,g,sub(rr,ii),0.,ui)
        ## Don't need to check for singularity in potrf! result b/c adding I
        potrs!('L',potrf!('L',syrk!('L','N',1.,g,1.,copy!(L[i],ee)))[1],ui)
    end
    pp
end

## Determine the conditional mode of the random effects using penalized nonlinear least squares
function pnlsfit!{T<:FP}(pp::SimplePopPK{T})
    m = pp.m; id = pp.inds; RX = pp.RX; lam = pp.lambda; L = pp.L; u = pp.u; delu = pp.delu
    oldprss = prss!(pp,0.)
    for i in 1:30
        # fill out the L arrays
    end
end
