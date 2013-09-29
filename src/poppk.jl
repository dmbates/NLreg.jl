type PopPK
    m::NLregMod{Float64}
    inds::Vector
    T::AbstractMatrix
    X::Matrix{Float64}
    Xs::Matrix{Float64}
    RX::Cholesky{Float64}
    L::Vector{Matrix{Float64}}
    beta::Vector{Float64}
    u::Matrix{Float64}
end
function PopPK(m::NLregMod,inds::Vector,T::AbstractMatrix,beta::Vector)
    p,n = size(m); np = n*p; ui = unique(inds); n1 = length(ui)
    isperm(ui) || error("unique(inds) should be a permutation")
    length(inds) == n || error("length(inds) = $(length(inds)) should be $n")
    size(T) == (p,p) || error("size(T) = $(size(T)) should be $((p,p))")
    PopPK(m,inds,T,X,Xs,cholfact(eye(size(X,2))),cholfact(LambdatZt,1.,true),
          LambdatZt,zeros(size(X,2)))
end
