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
    u::Matrix{T}
    u₀::Matrix{T}
    δu::Matrix{T}
    λ::LowerTriangular{T,Matrix{T}}
    pars::Vector{T}                  # parameter vector for evaluation of f
    L11::Vector{Matrix{T}}
    L21::Matrix{T}
    L22::Matrix{T}
    reinds::Vector{Int}
    residuals::Vector{Vector{T}}
    data::GroupedDataFrame
    Rdf::Ref                         # Ref to a subDataFrame of data
    ynm::Symbol
    res::Vector                      # vector of DiffResults.MutableDiffResult
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
    b = zeros(T, j, m)
    NLmixedModel(f, pnms, φ, copy(φ), zeros(T, k), b, copy(b), zeros(T, size(b)), λ, copy(φ),
        L11, L21, L22, reindsu, residuals, data, Rdf, ynm, res, cfg)
end

NLmixedModel(m::Function, d::GroupedDataFrame, nm::Symbol, β::NamedTuple) = NLmixedModel(m, d, nm, β, collect(1:length(β)))

"""
    fullL(mod::NLmixedModel)

Return the full lower triangular `Λ`, and the full lower Cholesky factor, `L`

Used for checking the values calculated in `updateL`.
"""
function fullL(mod::NLmixedModel{N,T}) where {N,T}
    L11 = mod.L11
    L21 = mod.L21
    m, n = size(L21)
    valsz = m + n
    val = zeros(T, valsz, valsz)
    k = size(first(L11), 1)
    Xcols = (1:N) .+ n
    XtX = view(val, Xcols, Xcols)
    rtX = view(val, size(val, 1), Xcols)
    cols = 1:k
    for (r, resid) in zip(mod.res, mod.residuals)
        jacobian = r.derivs[1]
        dblk = view(val, cols, cols)          # diagonal block
        mul!(dblk, jacobian', jacobian)       # N.B. only works if mod.reinds == 1:k
        copyto!(view(val, Xcols, cols), dblk)
        mul!(view(val, size(val, 1), cols), jacobian', resid)
        BLAS.syrk!('L', 'T', one(T), jacobian, one(T), XtX)
        BLAS.gemv!('T', one(T), jacobian, resid, one(T), rtX)
        cols = cols .+ k
    end
    val[end, end] = 
    LinearAlgebra.copytri!(val, 'L')
    Λ = LowerTriangular(kron(Diagonal(ones(length(L11))), mod.λ))
    lmul!(Λ', view(val, 1:n, :))
    rmul!(view(val, :, 1:n), Λ)
    for j in 1:n
        val[j, j] += 1
    end
    u = mod.u
    for j in eachindex(u)
        val[valsz, j] -= u[j]
    end
    A = Symmetric(val, :L)
    Λ, A, cholesky(A).L
end

"""
    dispersion(m::NLmixedModel, sqr::Bool=false)

Return the estimate of the dispersion parameter, σ, or its square, if `sqr` is `true`.
Note: this is the maximum likelihood estimate.  The denominator is the number of observations.
"""
function dispersion(m::NLmixedModel, sqr::Bool=false)
    resid = m.residuals
    val = (sum(abs2, m.u) + sum(r -> sum(abs2, r), resid)) / sum(length, resid)
    sqr ? val : sqrt(val)
end

dof_residual(m::NLmixedModel) = sum(length, m.residuals)  # we're using mle's

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

Overwrite the `δ` and `δu` fields on `mod` with the current increments from `L21` and `L22`.
"""
function increment!(mod::NLmixedModel{N,T}) where {N,T}
    δ = mod.δ
    δu = mod.δu
    L11 = mod.L11
    L21 = mod.L21
    L22 = mod.L22
    m, n = size(L21)
    copyto!(δ, view(L22, m, 1:N))
    ldiv!(LowerTriangular(view(mod.L22, 1:N, 1:N))', δ)
    mul!(vec(δu), view(L21, 1:N, :)', δ)
    for i in eachindex(δu)
        δu[i] = L21[m, i] - δu[i]
    end
    for (j, L) in enumerate(L11)
        ldiv!(LowerTriangular(L)', view(δu, :, j))
    end
    mod
end

function LinearAlgebra.logdet(m::NLmixedModel)
    L11 = m.L11
    dind = diagind(first(L11))
    2 * sum(L -> sum(log(L[j]) for j in dind), L11)
end 

"""
    objective(m::NLmixedModel)

Return the Laplace approximation to negative twice the log-likelihood.  Assumes `m.res`,
`m.residuals` and `m.L11` have been updated (via `updateμ!` and `updateL`) to the current
`m.φ` and `m.u`.
"""
function objective(m::NLmixedModel)
    L11 = m.L11
    dind = diagind(first(L11))
    logdet(m) + dof_residual(m)*(1 + log(2π * dispersion(m, true)))
end

"""
    pnls!(m::NLmixedModel; verbose=false, tol=1.0e-9, minstep=0.001, maxiter=500)

Optimize the penalized residual sum of squares w.r.t `m.φ` and `m.u`
"""
function pnls!(m::NLmixedModel; verbose=false, tol=1.0e-9, minstep=0.001, maxiter=500)
    u = fill!(m.u, 0)
    u₀ = fill!(m.u₀, 0)
    oldprss = updateμ!(m)
    cvg = updateL!(m)
    increment!(m)
    φ = m.φ
    φ₀ = copyto!(m.φ₀, φ)
    verbose && @show cvg, oldprss, φ
    iter = 1
    while cvg > tol && iter ≤ maxiter
        step = 1.0                              # step factor
        prss = updateφu!(m, step)
        while prss > oldprss && step ≥ minstep  # step-halving to ensure reduction of prss
            step /= 2
            prss = updateφu!(m, step)
        end
        if step < minstep
            throw(ErrorException("Step factor reduced below minstep of $minstep"))
        end
        copyto!(φ₀, φ)
        copyto!(u₀, u)
        cvg = updateL!(m)
        increment!(m)
        iter += 1
        oldprss = prss
        verbose && @show cvg, oldprss, φ
    end
    if iter > maxiter
        throw(ErrorException("Maximum number of iterations, $maxiter, exceeded"))
    end
    updateφu!(m, 1.0)
    updateL!(m)
    increment!(m)
    m
end

Base.propertynames(mod::NLmixedModel) = push!(collect(fieldnames(NLmixedModel)), :θ)

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

function updateφu!(m::NLmixedModel, step)
    @. m.φ = m.φ₀ + step * m.δ
    @. m.u = m.u₀ + step * m.δu
    updateμ!(m)
end

"""
    updateμ!(mod::NLmixedModel)

Update `mod.res`, containing the fitted values and the Jacobians, and `mod.residuals`.
Returns the penalized residual sum of squares.
"""
function updateμ!(mod::NLmixedModel)
    f = mod.f
    φ = mod.φ
    u = mod.u
    ynm = mod.ynm
    cfg = mod.cfg
    reinds = mod.reinds
    λ = mod.λ
    pars = mod.pars
    prss = sum(abs2, u)
    for (r, sdf, resid, j) in zip(mod.res, mod.data, mod.residuals, eachindex(mod.res))
        mul!(pars, λ, view(u, :, j))
        @. pars += φ
        mod.Rdf[] = sdf
        ForwardDiff.jacobian!(r, f, pars, cfg)
        @. resid = sdf[ynm] - r.value
        prss += sum(abs2, resid)
    end
    prss
end

"""
    updateL!(mod::NLmixedModel)

Update `mod.L11`, `mod.L21`, and `mod.L22`, returning the convergence criterion.
"""
function updateL!(mod::NLmixedModel{N,T}) where {N,T}
    u = mod.u
    λ = mod.λ
    L21 = fill!(mod.L21, zero(T))
    L22 = fill!(mod.L22, zero(T))
    m, n = size(L21)
    L11 = mod.L11
    nre = size(λ, 1)
    dind = diagind(first(L11))
    XtX = view(L22, 1:N, 1:N)
    rtX = view(L22, m, 1:N)
    cols = 1:nre
    for (r, df, L, resid, j) in zip(mod.res, mod.data, mod.L11, mod.residuals, eachindex(mod.res))
        rd = r.derivs[1]
        BLAS.syrk!('L', 'T', one(T), rd, one(T), XtX)
        BLAS.gemv!('T', one(T), rd, resid, one(T), rtX)
        lmul!(λ', rmul!(mul!(L, rd', rd), λ))
        for i in dind
            L[i] += one(T)
        end
        ch = cholesky!(Symmetric(L, :L)).L
        mul!(view(L21, 1:N, cols), rd', rd)
        allrows = view(L21, :, cols)
        mul!(view(L21, N+1, cols), rd', resid)
        rmul!(allrows, λ)
        for (i, c) in enumerate(cols)
            L21[N+1, c] -= u[i,j]
        end
        rdiv!(allrows, ch')
        cols = cols .+ nre
    end
    L22[end,end] = sum(abs2, u) + sum(r -> sum(abs2, r), mod.residuals)
    BLAS.syrk!('L', 'N', -one(T), L21, one(T), L22)
    cholesky!(Symmetric(L22, :L))
    (sum(abs2, view(L21, m, :)) + sum(abs2, view(L22, m, 1:N))) / abs2(L22[m,m])
end
