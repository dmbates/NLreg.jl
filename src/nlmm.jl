abstract NLMM{T<:FP} <: StatisticalModel

StatsBase.model_response(nm::NLMM) = model_response(nlregmod(nm))
StatsBase.residuals(nm::NLMM) = residuals(nlregmod(nm))
pnames(nm::NLMM) = pnames(nlregmod(nm))
npars(nm::NLMM) = npars(nlregmod(nm))
StatsBase.df_residual(nm::NLMM) = nobs(nm) - npars(nm)

## returns the coefficient table
function StatsBase.coeftable(nm::NLMM)
    pp = coef(nm); se = stderr(nm); tt = pp ./ se
    CoefTable(hcat(pp, se, tt, ccdf(FDist(1, df_residual(nm)), abs2(tt)))
              ["Estimate","Std.Error","t value", "Pr(>|t|)"],
              pnames(nm), 4)
end

theta(nm::NLMM) = theta(nm.lambda)
theta(lambda::Diagonal) = lambda.diag
function theta(m::Triangular)
    n = size(m,1); v = Array(eltype(m),n*(n+1)>>1); pos = 0; UL = m.UL
    for j in 1:n, i in (m.uplo == 'L' ? (j:n) : (1:j)) v[pos += 1] = UL[i,j] end
    v
end

theta!{T}(lambda::Diagonal{T},th::Vector{T}) = copy!(lambda.diag,th)
function theta!{T}(m::Triangular{T},th::Vector{T})
    n = size(m,1); pos = 0; UL = m.UL
    length(th) == n*(n+1)>>1 || error("length(th) = $(length(th)), should be $(n*(n+1)/2)")
    for j in 1:n, i in (m.uplo == 'L' ? (j:n) : (1:j)) UL[i,j] = th[pos += 1] end
    m
end

lowerbd{T<:FP}(nm::NLMM{T}) = [fill(-inf(T),length(nm.beta)),lowerbd(nm.lambda)]
lowerbd{T}(lambda::Diagonal{T}) = zeros(T,size(lambda,1))
function lowerbd{T}(m::Triangular{T})
    n = size(m,1); v = fill(-inf(T),n*(n+1)>>1); pos = 1; zt = zero(T)
    for j in 1:n, i in (m.uplo == 'L' ? (j:n) : (1:j))
        i == j && (v[pos] = zt)
        pos += 1
    end
    v
end

function setpars!{T<:FP}(nm::NLMM{T},pars::Vector{T})
    bb = nm.beta; p = length(bb); copy!(bb, view(pars,1:p))
    theta!(nm.lambda,pars[(p+1):end])
    nm
end
    
function StatsBase.fit(nm::NLMM; verbose=false)
    pnls!(nm; verbose=verbose)
    u0 = copy(nm.u)
    th = theta(nm); nth = length(th)
    pars = [nm.beta,th]
    o = Opt(:LN_BOBYQA,length(pars))
    ftol_abs!(o, 1e-6)    # criterion on deviance changes
    xtol_abs!(o, 1e-6)    # criterion on all parameter value changes
    lower_bounds!(o, lowerbd(nm))
    function obj(x::Vector{Float64}, g::Vector{Float64})
        length(g) == 0 || error("gradient evaluations are not provided")
        pnls!(setpars!(nm,x),u0;verbose=verbose)
    end
    if verbose
        count = 0
        function vobj(x::Vector{Float64}, g::Vector{Float64})
            count += 1
            length(g) == 0 || error("gradient evaluations are not provided")
            val = pnls!(setpars!(nm,x),u0)
            print("f_$count: $(round(val,5)), [")
            showcompact(x[1])
            for i in 2:length(x) print(","); showcompact(x[i]) end
            println("]")
            val
        end
        min_objective!(o, vobj)
    else
        min_objective!(o, obj)
    end
    fmin, xmin, ret = optimize!(o, pars)
    verbose && println(ret)
    setpars!(nm,xmin)
    nm
end

function Base.scale(nm::NLMM, sqr=false)
    ssqr = pwrss(nm)/nobs(nm)
    sqr ? ssqr : sqrt(ssqr)
end

StatsBase.vcov(nm::NLMM) = scale(nm,true) * unscaledvcov(nlregmod(nm))
    
