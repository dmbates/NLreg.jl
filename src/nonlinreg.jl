struct NLregModel{N,T<:AbstractFloat} <: StatisticalModel
    f::Function
    pnms::NTuple{N,Symbol}
    φ::Vector{T}
    y::Vector{T}
    R::UpperTriangular{T,Matrix{T}}
    data::AbstractDataFrame
    res::MutableDiffResult
    cfg::JacobianConfig
end

function NLregModel(model::Function, data, β::NamedTuple, ynm::Symbol=:y)
    φ = collect(β)
    y = convert(typeof(φ), data[ynm])
    n = length(y)
    if size(data, 1) ≠ n
        throw(ArgumentError("size(data, 1) = $(size(data, 1)) ≠ length(y) = $n"))
    end
    k = length(β)
    f(φ) = model(φ, data)
    NLregModel(f, keys(β), φ, y, UpperTriangular(similar(φ, k, k)), data,
        JacobianResult(y,φ), JacobianConfig(f,φ))
end

"""
    decrement!(δ, res, y)

Evaluates the negative of the increment (i.e. the decrement) in `δ` from
the negative residual and the Jacobian in `res`, which is overwritten in
the process.  Returns the residual sum of squares and the convergence
criterion.
"""
function decrement!(δ, res)
    negativeresid = res.value
    rss = sum(abs2, negativeresid)
    qrfac = qr!(res.derivs[1])
    lmul!(qrfac.Q', negativeresid)
    copyto!(δ, 1, negativeresid, 1, length(δ))
    cvg = sum(abs2, δ) / rss
    ldiv!(UpperTriangular(qrfac.R), δ)
    rss, sqrt(cvg)
end

function fit!(m::NLregModel; verbose=false, tol=0.00001, minstep=0.001, maxiter=100)
    data = m.data
    φ = m.φ
    δ = similar(φ)
    f = m.f
    y = m.y
    cfg = m.cfg
    res = m.res
    jacobian!(m.res, f, φ, cfg).value .-= y   # evaluate negative residuals and Jacobian
    oldrss, cvg = decrement!(δ, res)
    verbose && @show cvg, oldrss, φ
    trialpars = similar(δ)
    iter = 1
    while cvg > tol && iter ≤ maxiter
        step = 1.0                            # step factor
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

Base.size(m::NLregModel) = size(m.res.derivs[1])
coef(m::NLregModel) = m.φ
coefnames(m::NLregModel) = m.pnms
function coeftable(m::NLregModel)
    φ = m.φ
    σ = stderror(m)
    CoefTable([φ, σ, φ ./ σ], ["Estimate","Std.Error","t value"],
        collect(String.(m.pnms)), 0, 3)
end
deviance(m::NLregModel) = rss(m)
dof(m::NLregModel{K}) where {K} = K
dof_residual(m::NLregModel{K}) where {K} = length(m.y) - K
fitted(m::NLregModel) = m.res.value
islinear(m::NLregModel) = false
function loglikelihood(m::NLregModel)
    n = nobs(m)
    -n * (1. + log(2π * rss(m)/n)) / 2
end
nobs(m::NLregModel) = length(m.y)
params(m::NLregModel{K,T}) where {K,T} = NamedTuple{m.pnms, NTuple{K,T}}(m.φ)
residuals(m::NLregModel) = m.y .- fitted(m)
response(m::NLregModel) = m.y
rss(m::NLregModel) = sum(abs2, m.y .- fitted(m))
function Base.show(io::IO, m::NLregModel)
    println(io, "Nonlinear regression model (NLregModel)")
    println(io)
    println(io, " Residual sum of squares: ", deviance(m))
    println(io)
    println(io, coeftable(m))
end

function vcov(m::NLregModel)
    Rinv = inv(m.R)
    Rinv * Rinv' .* (rss(m)/dof_residual(m))
end

#=
abstract PLregModF{T<:FP} <: NLregModF{T}

## default methods for all PLregModF objects
Base.size(m::PLregModF) = size(mmjac(m))
Base.size(m::PLregModF,args...) = size(mmjac(m),args...)
StatsBase.model_response(m::PLregModF) = m.y
mmfunc(m::PLregModF) = m.mmf # model-matrix update function
mmjac(m::PLregModF) = m.MMD # model-matrix derivative (Jacobian)

## update the (transposed) model matrix for the linear parameters and the gradient MMD
function updtMM!(m::PLregModF,nlpars)
    mmd = mmjac(m)
    mmf = mmfunc(m)
    tg  = tgrad(m)
    x   = covariatemat(m)
    y   = model_response(m)
    for i in 1:size(x,2)
        mmf(nlpars,view(x,:,i),view(tg,:,i),view(mmd,:,:,i))
    end
    tg[1:size(m,2),:]
end

## update mu and resid from full parameter vector returning rss
function updtmu!(m::PLregModF,pars::Vector)
    length(pars) == npars(m) || throw(DimensionMismatch(""))
    mmd = mmjac(m)
    mmf = mmfunc(m)
    mu  = expctd(m)
    r   = residuals(m)
    tg  = tgrad(m)
    x   = covariatemat(m)
    y   = model_response(m)
    nnl,nl,n = size(m.MMD); lind = 1:nl; nlind = nl + (1:nnl)
    nlpars = view(pars,nlind); lpars = view(pars,lind)
    rss = zero(eltype(mu))
    for i in 1:n
        mmdi = view(mmd,:,:,i)
        mmf(nlpars,view(x,:,i),view(tg,:,i),mmdi)
        tg[nlind,i] = mmdi * lpars
        r[i] = y[i] - (mu[i] = dot(view(tg,lind,i),lpars))
        rss += abs2(r[i])
    end
    rss
end

## update mu and resid from parameter matrix and indicators returning rss
function updtmu!(m::PLregModF,pars::Matrix,inds::Vector)
    nnl,nl,n = size(m); lind = 1:nl; nlind = nl + (1:nnl)
    length(inds) == n && size(pars,1) == npars(m) || throw(DimensionMismatch(""))
    mmd = mmjac(m)
    mmf = mmfunc(m)
    mu  = expctd(m)
    r   = residuals(m)
    tg  = tgrad(m)
    x   = covariatemat(m)
    y   = model_response(m)
    nnl,nl,n = size(m); lind = 1:nl; nlind = nl + (1:nnl)
    ii  = 0
    rss = zero(eltype(mu))
    for i in 1:n
        if ii != inds[i]
            ii = inds[i]
            nlp = view(pars,nlind,ii)
            lp = view(pars,lind,ii)
        end
        mmdi = view(mmd,:,:,i)
        mmf(nlp,view(x,:,i),view(tg,:,i),mmdi)
        tg[nlind,i] = mmdi * lp
        r[i] = y[i] - (mu[i] = dot(view(tg,lind,i),lp))
        rss += abs2(r[i])
    end
    rss
end
=#
