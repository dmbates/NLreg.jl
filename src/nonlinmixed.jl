using BlockArrays, CSV, DataFrames, ForwardDiff, LinearAlgebra
using DiffResults: MutableDiffResult
using ForwardDiff: JacobianConfig
using StatsBase: StatisticalModel

struct NLmixedModel{N,T<:AbstractFloat} <: StatisticalModel
    f::Function
    pnms::NTuple{N,Symbol}
    φ::Vector{T}
    b::Matrix{T}
    λ::LowerTriangular{T,Matrix{T}}
    L11::Vector{Matrix{T}}
    L21::Matrix{T}
    L22::Matrix{T}
    reinds::Vector{Int}
    data::GroupedDataFrame
    Rdf::Ref
    ynm::Symbol
    res::Vector
    cfg::JacobianConfig
end

function NLmixedModel(model::Function, data::GroupedDataFrame, ynm::Symbol,
    β::NamedTuple, reinds::Vector{Int})
    φ = float(collect(β))
    T = eltype(φ)
    Rdf = Ref(first(data))
    f(x) = model(x, Rdf[])
    res = [DiffResults.JacobianResult(sdf[ynm], φ) for sdf in data]
    cfg = ForwardDiff.JacobianConfig(f, φ)
    for (r, sdf) in zip(res, data)
        Rdf[] = sdf
        ForwardDiff.jacobian!(r, f, φ, cfg)
    end
    pnms = keys(β)
    k = length(pnms)
    reindsu = unique(reinds)
    all(1 .≤ reindsu .≤ k) || throw(ArgumentError("all reinds must be in [1,$k]"))
    j = length(reindsu)
    m = length(data)
    λ = LowerTriangular(zeros(T, j, j) + I)
    L11 = [zeros(T, j, j) for i in 1:m]
    L21 = zeros(T, k+1, j*m)
    L22 = zeros(T, k+1, k+1)
    b = zeros(j, m)
    NLmixedModel(f, pnms, φ, b, λ, L11, L21, L22, reindsu, data, Rdf, ynm, res, cfg)
end

NLmixedModel(m::Function, d::GroupedDataFrame, nm::Symbol, β::NamedTuple) = 
    NLmixedModel(m, d, nm, β, collect(1:length(β)))

function updateμ!(mod::NLmixedModel)
    f = mod.f
    φ = mod.φ
    cfg = mod.cfg
    for (r, sdf) in zip(mod.res, mod.data)
        mod.Rdf[] = sdf
        ForwardDiff.jacobian!(r, f, φ, cfg)
    end
    mod
end

function updateL!(mod::NLmixedModel{N,T}) where {N,T}
    λ = mod.λ
    L21 = fill!(mod.L21, zero(T))
    L22 = fill!(mod.L22, zero(T))
    ngrps = length(mod.data)
    m, n = size(L21)
    L11 = mod.L11
    nre = size(first(L11), 1)
    dind = diagind(first(L11))
    XtX = view(L22, 1:N, 1:N)
    rtX = view(L22, m, 1:N)
    rss = zero(T)
    cols = 1:nre
    for (r, df, L) in zip(mod.res, mod.data, mod.L11)
        rv = r.value
        map!(-, rv, df[mod.ynm], rv) # evaluate residual in r.value
        rss += sum(abs2, rv)
        rd = r.derivs[1]
        BLAS.syrk!('L', 'T', one(T), rd, one(T), XtX)
        BLAS.gemv!('T', one(T), rd, rv, one(T), rtX)
        lmul!(λ', rmul!(mul!(L, rd', rd), λ))
        for i in dind
            L[i] += one(T)
        end
        ch = cholesky!(Symmetric(L, :L)).L
        mul!(view(L21, 1:N, cols), rd', rd)
        mul!(view(L21, N+1, cols), rd', rv)
        rdiv!(rmul!(view(L21, :, cols), λ), ch')
        cols = cols .+ nre
    end
    L22[end,end] = rss
    BLAS.syrk!('L', 'N', -one(T), L21, one(T), L22)
    cholesky!(Symmetric(L22, :L))
    mod
end

function fullL(mod::NLmixedModel{N,T}) where {N,T}
    L11 = mod.L11
    L21 = mod.L21
    m, n = size(L21)
    val = zeros(T, m + n, m + n)
    k = size(first(L11), 1)
    Xcols = (1:N) .+ n
    XtX = view(val, Xcols, Xcols)
    rtX = view(val, size(val, 1), Xcols)
    rss = zero(T)
    cols = 1:k
    for (r, df) in zip(mod.res, mod.data)
        resid = map!(-, r.value, df[mod.ynm], r.value)
        jacobian = r.derivs[1]
        dblk = view(val, cols, cols)  # diagonal block
        mul!(dblk, jacobian', jacobian)
        copyto!(view(val, Xcols, cols), dblk)
        lblk = view(val, size(val, 1), cols)
        mul!(lblk, jacobian', resid)
        BLAS.syrk!('L', 'T', one(T), jacobian, one(T), XtX)
        BLAS.gemv!('T', one(T), jacobian, resid, one(T), rtX)
        rss += sum(abs2, resid)
        cols = cols .+ k
    end
    LinearAlgebra.copytri!(val, 'L')
    val[end, end] = rss
    Λ = LowerTriangular(kron(Diagonal(ones(length(L11))), mod.λ))
    lmul!(Λ', view(val, 1:n, :))
    rmul!(view(val, :, 1:n), Λ)
    for j in 1:n
        val[j, j] += 1
    end
    Λ, Symmetric(val, :L)
end

