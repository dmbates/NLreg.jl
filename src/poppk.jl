type SimplePopPK{T<:FP} <: StatisticalModel
    m::NLregMod{T}
    inds::Vector
    lambda::AbstractMatrix{T}
    RX::Cholesky{T}
    L::Vector{Matrix{T}}  # should this be a 3-D array?
    beta::Vector{T}       # fixed-effects parameter vector
    u::Matrix{T}          # spherical random-effects values
    delu::Matrix{T}       # increment in the PNLS algorithm
    phi::Matrix{T}      # parameter matrix - phi := beta + lambda * (u + fac*delu)
end
function SimplePopPK{T<:FP}(m::NLregMod{T},inds::Vector,lambda::AbstractMatrix{T},beta::Vector{T})
    p,n = size(m); np = n*p; ui = unique(inds); ni = length(ui)
    isperm(ui) || error("unique(inds) should be a permutation")
    length(inds) == n || error("length(inds) = $(length(inds)) should be $n")
    size(lambda) == (p,p) || error("size(lambda) = $(size(lambda)) should be $((p,p))")
    L = Matrix{T}[eye(T,p) for i in 1:ni]
    u = zeros(T,(p,ni))
    SimplePopPK(m,inds,lambda,cholfact(eye(T,p)),L,beta,u,copy(u),similar(u))
end
## ToDo Add an external constructor using Formula/Data

## Multiply u by lambda in place creating the b values
u2b!{T<:FP}(lambda::Diagonal{T},u::Matrix{T}) = scale!(lambda.diag,u)
u2b!{T<:FP}(lambda::Triangular{T},u::Matrix{T}) = trmm!('L','L','N','N',1.,lambda,u)

## penalized residual sum of squares
function prss!{T<:FP}(pp::SimplePopPK{T},fac::T)
    phi = pp.phi
    copy!(phi,pp.u)
    fma!(phi,pp.delu,fac)             # evaluate u + fac*delu in phi
    ssu = sqsum(phi)
    broadcast!(+,u2b!(pp.lambda,phi),pp.beta) # phi := beta + lambda * (u + fac*delu)
    updtmu!(pp.m,phi,pp.inds) + ssu
end

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

## Determine the conditional mode of the random effects using penalized nonlinear least squares
function pnlsfit!{T<:FP}(pp::SimplePopPK{T})
    m = pp.m; id = pp.inds; RX = pp.RX; lam = pp.lambda; L = pp.L; u = pp.u; delu = pp.delu
    oldprss = prss!(pp,0.)
    for i in 1:30
        # fill out the L arrays
    end
end
