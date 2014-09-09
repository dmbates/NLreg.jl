abstract NLregModF{T<:FP}

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
