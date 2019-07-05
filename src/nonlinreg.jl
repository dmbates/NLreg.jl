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

function NLregModel(model, data, y, β::NamedTuple)
    n = length(y)
    if size(data, 1) ≠ n
        throw(ArgumentError("size(data, 1) = $(size(data, 1)) ≠ length(y) = $n"))
    end
    φ = collect(β)
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
    jacobian!(m.res, f, φ, cfg).value .-= y # evaluate negative residual and Jacobian
    res.value .-= y
    oldrss, cvg = decrement!(δ, res)
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

Base.size(m::NLregModel) = size(m.res.derivs[1])
fitted(m::NLregModel) = m.res.value
residuals(m::NLregModel) = m.y - fitted(m)
params(m::NLregModel{K,T}) where {K,T} = NamedTuple{m.pnms, NTuple{K,T}}(m.φ)
#=
abstract type NLregModF{T<:AbstractFloat}

## default methods for all NLregModF objects
Base.size(m::NLregModF) = size(tgrad(m))
Base.size(m::NLregModF,args...) = size(tgrad(m),args...)
StatsBase.model_response(m::NLregModF) = m.y
StatsBase.nobs(m::NLregModF) = length(model_response(m))
StatsBase.residuals(m::NLregModF) = m.resid
covariatemat(m::NLregModF) = m.x
expctd(m::NLregModF) = m.mu
mufunc(m::NLregModF) = m.f
npars(m::NLregModF) = size(tgrad(m),1)
tgrad(m::NLregModF) = m.tgrad # transposed gradient matrix
unscaledvcov(m::NLregModF) = inv(cholfact(tgrad(m) * tgrad(m)'))

## update the expected response and residuals, returning the sum of squared residuals
function updtmu!(m::NLregModF, pars::Vector)
    length(pars) == npars(m) || throw(DimensionMismatch(""))
    f  = mufunc(m)
    mu = expctd(mu)
    r  = residuals(m)
    tg = tgrad(m)
    x  = covariatemat(m)
    y  = model_response(m)
    for i in 1:length(y)
        mu[i] = f(pars,view(x,:,i),view(tgrad,:,i)) # pass subarrays by reference, not copies
        r[i] = y[i] - mu[i]
        rss += abs2(r[i])
    end
    rss
end

function updtmu!(m::NLregModF, pars::Matrix, inds::Vector)
    (n == nobs(m)) == length(inds) && size(pars,1) == npars(m) || throw(DimensionMismatch(""))
    f  = mufunc(m)
    mu = expctd(mu)
    r  = residuals(m)
    tg = tgrad(m)
    x  = covariatemat(m)
    y  = model_response(m)
    ii = 0; rss = zero(eltype(mu))
    for i in 1:n
        if ii != inds[i]
            ii = inds[i]
            pp = view(pars,:,ii)
        end
        mu[i] = f(pp,view(x,:,i),view(tgrad,:,i))
        r[i]  = y[i] - mu[i]
        rss  += abs2(r[i])
    end
    rss
end

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
