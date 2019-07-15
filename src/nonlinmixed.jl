using DataFrames, ForwardDiff, LinearAlgebra
using ForwardDiff.DiffResults: MutableDiffResult
using ForwardDiff: JacobianConfig
using StatsBase: StatisticalModel

struct NLmixedModel{N,T<:AbstractFloat} <: StatisticalModel
    f::Function
    pnms::NTuple{N,Symbol}
    φ::Vector{T}
    φ₀::Vector{T}
    δ::Vector{T}
    b::Matrix{T}
    b₀::Matrix{T}
    δb::Matrix{T}
    λ::LowerTriangular{T,Matrix{T}}
    pars::Vector{T}
    L11::Vector{Matrix{T}}
    L21::Matrix{T}
    L22::Matrix{T}
    reinds::Vector{Int}
    residuals::Vector{Vector{T}}
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
    residuals = [copy(r.value) for r in res]
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
    NLmixedModel(f, pnms, φ, copy(φ), similar(φ), b, copy(b), similar(b), λ, copy(φ),
        L11, L21, L22, reindsu, residuals, data, Rdf, ynm, res, cfg)
end

NLmixedModel(m::Function, d::GroupedDataFrame, nm::Symbol, β::NamedTuple) = NLmixedModel(m, d, nm, β, collect(1:length(β)))

"""
    fullL(mod::NLmixedModel)

Update `mod.res` and `mod.residuals` and return the full lower triangular `Λ` and the full `A`

Used for checking the values calculated in `updateL`.
"""
function fullL(mod::NLmixedModel{N,T}) where {N,T}
    rss = updateμ!(mod)
    L11 = mod.L11
    L21 = mod.L21
    m, n = size(L21)
    val = zeros(T, m + n, m + n)
    k = size(first(L11), 1)
    Xcols = (1:N) .+ n
    XtX = view(val, Xcols, Xcols)
    rtX = view(val, size(val, 1), Xcols)
    cols = 1:k
    for (r, df, resid) in zip(mod.res, mod.data, mod.residuals)
        jacobian = r.derivs[1]
        dblk = view(val, cols, cols)  # diagonal block
        mul!(dblk, jacobian', jacobian)
        copyto!(view(val, Xcols, cols), dblk)
        lblk = view(val, size(val, 1), cols)
        mul!(lblk, jacobian', resid)
        BLAS.syrk!('L', 'T', one(T), jacobian, one(T), XtX)
        BLAS.gemv!('T', one(T), jacobian, resid, one(T), rtX)
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

"""
    getθ!(v, λ::LowerTriangular)
    getθ!(v, mod::NLmixedModel)

Overwrite `v` with the elements (column-major ordering) of the lower triangle of `λ` (or `mod.λ`)
""" 
function getθ!(v::AbstractVector{T}, λ::LowerTriangular{T}) where {T}
    n = LinearAlgebra.checksquare(λ)
    if length(v) ≠ nlower(n)
        throw(DimensionMismatch("length(v) = $(length(v)) should be $(nlower(n))"))
    end
    mat = λ.data
    ind = 1
    for j in 1:n
        for i in j:n
            v[ind] = mat[i, j]
            ind += 1
        end
    end
    v
end

getθ!(v::AbstractVector{T}, mod::NLmixedModel{N,T}) where {N,T} = getθ!(v, mod.λ)

"""
    getθ(λ::LowerTriangular)
    getθ(mod::NLmixedModel)

Return a vector of the elements (column-major ordering) of the lower triangle of `λ` (or `mod.λ`)
""" 
getθ(λ::LowerTriangular{T}) where {T} = 
    getθ!(Vector{T}(undef, nlower(LinearAlgebra.checksquare(λ))), λ)

getθ(mod::NLmixedModel) = getθ(mod.λ)

Base.getproperty(mod::NLmixedModel, s::Symbol) = s == :θ ? getθ(mod) : getfield(mod, s)

"""
    increment!(mod::NLmixedModel)

Overwrite the `δ` and `δb` fields on `mod` with the current increments from `L21` and `L22`.
"""
function increment!(mod::NLmixedModel{N,T}) where {N,T}
    δ = mod.δ
    δb = mod.δb
    L11 = mod.L11
    L21 = mod.L21
    L22 = mod.L22
    λ = mod.λ
    ngrps = length(mod.L11)
    nre = size(λ, 1)
    m, n = size(L21)
    copyto!(δ, view(L22, m, 1:N))
    ldiv!(LowerTriangular(view(mod.L22, 1:N, 1:N)), δ)
    mul!(vec(δb), view(L21, 1:N, :)', δ)
    for i in eachindex(δb)
        δb[i] = L21[m, i] - δb[i]
    end
    for (j, L) in enumerate(L11)
        lmul!(λ, ldiv!(LowerTriangular(L)', view(δb, :, j)))
    end
    mod
end

"""
    pnls(m::NLmixedModel; verbose=false, tol=0.00001, minstep=0.001, maxiter=500)

Optimize the penalized residual sum of squares w.r.t `m.φ` and `m.b`
"""
function pnls!(m::NLmixedModel; verbose=false, tol=0.00001, minstep=0.001, maxiter=500)
    cvg = updateL!(m)
    increment!(m)
    φ = m.φ
    φ₀ = copyto!(m.φ₀, φ)    
    b = m.b
    b₀ = copyto!(m.b₀, b)
    verbose && @show cvg, oldrss, φ
    trialpars = similar(δ)
    iter = 1
    while cvg > tol && iter ≤ maxiter
        step = 1.0                    # step factor
        @. trialpars = φ - step * δ
        jacobian!(res, f, trialpars, cfg).value .-= y
        rss = sum(abs2, res.value)
        while rss > oldrss && step ≥ minstep  # step-halving to ensure reduction of rss
            step /= 2
            @. trialpars = φ - step * δ
            jacobian!(res, f, trialpars, cfg).value .-= y
            rss = sum(abs2, res.value)
        end
        if step < minstep
            throw(ErrorException("Step factor reduced below minstep of $minstep"))
        end
        copyto!(φ, trialpars)
        rss, cvg = decrement!(δ, res)
        iter += 1
        oldrss = rss
        verbose && @show cvg, oldrss, φ
    end
    if iter > maxiter
        throw(ErrorException("Maximum number of iterations, $maxiter, exceeded"))
    end
    copyto!(m.R.data, view(res.derivs[1], 1:length(φ), :))
    jacobian!(res, f, φ, cfg)
    m
end

Base.propertynames(mod::NLmixedModel) = push!(collect(fieldnames(NLMixedModel)), :θ)

"""
    setθ!(λ::LowerTriangular, v)
    setθ!(mod::NLmixedModel, v)

Overwrite the lower-triangle of `λ` (or `mod.λ`) with the elements of `v`
""" 

function setθ!(λ::LowerTriangular{T}, v::AbstractVector{T}) where {T}
    n = LinearAlgebra.checksquare(λ)
    if length(v) ≠ nlower(n)
        throw(DimensionMismatch("length(v) = $(length(v)) should be $(nlower(n))"))
    end
    mat = λ.data
    ind = 1
    for j in 1:n
        for i in j:n
            mat[i, j] = v[ind]
            ind += 1
        end
    end
    λ
end

function setθ!(mod::NLmixedModel{N,T}, v::AbstractVector{T}) where {N,T}
    setθ!(mod.λ, v)
    mod
end

setproperty!(mod::NLregModel, s::Symbol, x) = s == :θ ? setθ!(mod, x) : setfield!(mod, s, x)

"""
    updateμ!(mod::NLmixedModel)

Update `mod.res`, containing the fitted values and the Jacobians, and `mod.residuals`
"""
function updateμ!(mod::NLmixedModel, step)
    f = mod.f
    φ = mod.φ
    b = mod.b
    ynm = mod.ynm
    cfg = mod.cfg
    pars = mod.pars
    reinds = mod.reinds
    rss = zero(eltype(φ))
    for (r, sdf, resid, j) in zip(mod.res, mod.data, mod.residuals, eachindex(mod.res))
        mod.Rdf[] = sdf
        copyto!(pars, φ)
        for (i, k) in enumerate(reinds)
            pars[k] += b[i, j]
        end
        ForwardDiff.jacobian!(r, f, pars, cfg)
        @. resid = sdf[ynm] - r.value
        rss += sum(abs2, resid)
    end
    rss
end

"""
    updateL!(mod::NLmixedModel)

Update `mod.L11`, `mod.L21`, and `mod.L22`, returning the convergence criterion.
"""
function updateL!(mod::NLmixedModel{N,T}) where {N,T}
    rss = updateμ!(mod)
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
    cols = 1:nre
    for (r, df, L, resid) in zip(mod.res, mod.data, mod.L11, mod.residuals)
        rd = r.derivs[1]
        BLAS.syrk!('L', 'T', one(T), rd, one(T), XtX)
        BLAS.gemv!('T', one(T), rd, resid, one(T), rtX)
        lmul!(λ', rmul!(mul!(L, rd', rd), λ))
        for i in dind
            L[i] += one(T)
        end
        ch = cholesky!(Symmetric(L, :L)).L
        mul!(view(L21, 1:N, cols), rd', rd)
        mul!(view(L21, N+1, cols), rd', resid)
        rdiv!(rmul!(view(L21, :, cols), λ), ch')
        cols = cols .+ nre
    end
    L22[end,end] = rss
    BLAS.syrk!('L', 'N', -one(T), L21, one(T), L22)
    cholesky!(Symmetric(L22, :L))
    (sum(abs2, view(L21, m, :)) + sum(abs2, view(L22, m, 1:N))) / abs2(L22[m,m])
end
