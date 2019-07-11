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
    L2::BlockArray{T,2,Matrix{T}}
    reinds::Vector{Int}
    data::GroupedDataFrame
    ynm::Symbol
    res::Vector
    cfg::JacobianConfig
end

function NLmixedModel(model::Function, data::GroupedDataFrame, ynm::Symbol,
    β::NamedTuple, reinds::Vector{Int})
    φ = float(collect(β))
    T = eltype(φ)
    global _sdf = data[1]
    f(x) = model(x, _sdf)
    res = [DiffResults.JacobianResult(sdf[ynm], φ) for sdf in data]
    cfg = ForwardDiff.JacobianConfig(f, φ)
    for (i, sdf) in enumerate(data)
        global _sdf = sdf
        ForwardDiff.jacobian!(res[i], f, φ, cfg)
    end
    pnms = keys(β)
    k = length(pnms)
    reindsu = unique(reinds)
    all(1 .≤ reindsu .≤ k) || throw(ArgumentError("all reinds must be in [1,$k]"))
    j = length(reindsu)
    m = length(data)
    λ = LowerTriangular(zeros(T, j, j) + I)
    L11 = [zeros(T, j, j) for i in 1:m]
    L2 = BlockArray{T}(undef, [k, 1], vcat(repeat([j], m), [k, 1]))
    b = zeros(j, m)
    NLmixedModel(f, pnms, φ, b, λ, L11, L2, reindsu, data, ynm, res, cfg)
end

function updateL!(m::NLmixedModel{N,T}) where {N,T}
    λ = m.λ
    L2 = m.L2
    k = length(m.data)
    XtX = fill!(L2[Block(1,k+1)], zero(T))
    rtX = fill!(L2[Block(2,k+1)], zero(T))
    rtr = fill!(L2[Block(2,k+2)], zero(T))
    for (r, df, L, j) in zip(m.res, m.data, m.L11, 1:k)
        rv = r.value
        map!(-, rv, df[m.ynm], rv) # evaluate residual in r.value
        rtr[1] += sum(abs2, rv)
        rd = r.derivs[1]
        BLAS.syrk!('L', 'T', one(T), rd, one(T), XtX)
        lmul!(λ', rmul!(mul!(L, rd', rd), λ))
        for i in diagind(L)
            L[i] += one(T)
        end
        ch = cholesky!(Symmetric(L, :L)).L
        rdiv!(rmul!(mul!(L2[Block(1,j)], rd', rd), λ), ch)
        rdiv!(rmul!(mul!(L2[Block(2,j)], rv', rd), λ), ch)
    end
    m
end
