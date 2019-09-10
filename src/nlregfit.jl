struct NLregModel{T} <: RegressionModel
    data::Vector{S} where {S<:NamedTuple}
    ysym::Symbol
    model::Function
    parref::Ref{R} where {R<:NamedTuple}
    current::Vector{T}
    chfac::Cholesky{T,Matrix{T}}
    R::UpperTriangular{T}
    incr::AbstractVector{T}
    scratch::Vector{T}
    optsum::Ref{U} where {U<:NamedTuple}
end

function NLregModel(data::Vector{<:NamedTuple}, ysym::Symbol, model::Function, pars::NamedTuple)
    if ysym ∉ keys(first(data))
        throw(ArgumentError("$ystm is not in the data row keys, $(keys(first(data)))"))
    end
    n = length(pars)
    T = float(first(promote_type(eltype.(values(pars)))))
    if !(values(pars) isa NTuple{n,T})
        pars = convert(NamedTuple{keys(pars),NTuple{n,T}}, values(pars))
    end
    current = collect(pars)
    np1 = n + 1
    chfac = cholesky!(zeros(T, np1, np1) + I)
    fac = chfac.factors
    NLregModel(
        data,
        ysym,
        model,
        Ref(pars),
        current,
        chfac,
        UpperTriangular(view(fac, 1:n, 1:n)),
        view(fac, 1:n, np1),
        similar(current, np1),
        Ref((rss=-one(T), cvg=-one(T), niter=-1))
    )
end

function updatech!(nlr::NLregModel)
    ch = nlr.chfac
    fac = ch.factors
    fill!(fac, false)
    pars = nlr.parref[]
    scr = nlr.scratch
    for r in nlr.data
        copyto!(scr, first(gradient(p -> nlr.model(p, r), pars)))
        scr[end] = getproperty(r, nlr.ysym) - nlr.model(pars, r)
        lowrankupdate!(ch, scr)
    end
    ssmod = sum(abs2, nlr.incr)
    ssres = abs2(fac[end, end])
    ldiv!(nlr.R, nlr.incr)
    ssmod + ssres, sqrt(ssmod / ssres)
end

struct IterationError <: Exception
    msg::String
end

function StatsBase.fit!(
    nls::NLregModel{T};
    maxiter = 200, tol = 1.0e-5, minfac = 1.0e-5, verbose = false) where {T}
    converged = false
    oldrss, cvg = updatech!(nls)
    parnms = keys(nls.parref[])
    current = nls.current
    incr = nls.incr
    iter = 1
    while iter ≤ maxiter
        if cvg < tol
            converged = true
            break
        end
        current .+= incr
        nls.parref[] = (; zip(parnms, current)...)
        rss, cvg = updatech!(nls)
        verbose && @show(i, rss, cvg)
        factor = 1.0
        copyto!(nls.scratch, current)  # keep a copy in case of step halving
        while oldrss < rss
            factor /= 2.
            if factor < minfac
                throw(IterationError("step factor = $factor less than minimum = $minfac"))
            end
            copyto!(current, 1, nls.scratch, 1, length(current))
            current .+= factor .* incr
            nls.parref[] = pars = (; zip(parnms, current)...)
            rss = sum(abs2(getproperty(r, nls.ysym) - nls.model(pars, r)) for r in nls.data)
            verbose && @show(factor, rss)
        end
        if !isone(factor)
            rss, cvg = updatech!(nls)
        end
        oldrss = rss
        iter += 1
    end
    if !converged
        throw(IterationError("maximum number of iterations, $maxiter, exceeded"))
    end
    nls.optsum[] = (rss = oldrss, cvg = cvg, niter = iter)
    nls
end

function StatsBase.fit(::Type{NLregModel}, data, ysym::Symbol, model::Function,
    pars::NamedTuple; kw...)
    fit!(NLregModel(Tables.rowtable(data), ysym, model, pars),
        verbose = get(kw, :verbose, false),
        minfac = get(kw, :minfac, 1.0e-5),
        maxiter = get(kw, :maxiter, 200),
        tol = get(kw, :tol, 1.0e-5)
    )
end

StatsBase.coef(nls::NLregModel) = nls.parref[]

StatsBase.coefnames(nls::NLregModel) = collect(string.(keys(nls.parref[])))

function StatsBase.coeftable(nls::NLregModel)
    cols = Any[]
    co = collect(nls.parref[])
    push!(cols, co)
    se = stderror(nls)
    push!(cols, se)
    push!(cols, co ./ se)
    CoefTable(cols, ["Estimate", "Std.Error", "t-statistic"],
        coefnames(nls))
end

StatsBase.dof_residual(nls::NLregModel) = nobs(nls) - length(nls.current)

StatsBase.islinear(::NLregModel) = false

function StatsBase.fitted(nls::NLregModel)
    pars = nls.parref[]
    [nls.model(pars, d) for d in nls.data]
end

StatsBase.nobs(nls::NLregModel) = length(nls.data)

function StatsBase.residuals(nls::NLregModel)
    pars = nls.parref[]
    [(getproperty(d, nls.ysym) - nls.model(pars, d)) for d in nls.data]
end

StatsBase.response(nls::NLregModel) = [getproperty(d, nls.ysym) for d in nls.data]

StatsBase.rss(nls::NLregModel) = nls.optsum[].rss

varest(nls::NLregModel) = rss(nls)/dof_residual(nls)

function StatsBase.vcov(nls::NLregModel)
    Rinv = inv(nls.R)
    varest(nls) * Rinv * Rinv'
end

function Base.show(io::IO, m::NLregModel)
    if m.optsum[].niter < 0
        @warn("Model has not been fit")
        return nothing
    end
    println(io, "Nonlinear regression model fit by maximum likelihood")
    println(io)
    println(io, "Data schema (response variable is ", m.ysym, ")")
    println(io, Tables.schema(m.data))
    println(io)
    println(io, " Sum of squared residuals at convergence: ", rss(m))
    println(io, " Achieved convergence criterion:          ", m.optsum[].cvg)
    println(io)
    println(io, " Number of observations:                  ", length(m.data))
    println(io)
    println(io, " Parameter estimates")
    show(io, coeftable(m))
end
