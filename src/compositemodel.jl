type CompositeModF{T<:FP} <: NLregModF{T}
    pt::Ptrans
    nm::NLregModF{T}
end

function (*)(vv::Vector{DataType},nm::NLregModF)
    dt = Dtrans(vv)
    length(dt) == npars(nm) || error("DimensionMisMatch")
    CompositeModF(dt,nm)
end

nlmod(cm::CompositeModF) = cm.nm

npars(cm::CompositeModF) = length(cm.pt)

ptran(cm::CompositeModF) = cm.pt

pnames(cm::CompositeModF) = pnames(ptran(cm), pnames(nlmod(cm)))

residuals(cm::CompositeModF) = residuals(cm.nm)

function updtmu!(cm::CompositeModF,pars::Vector)
    length(pars) == npars(cm) || error("DimensionMismatch")
    nm = nlmod(cm)
    tpars, jac = parjac(ptran(cm), pars)
    res = updtmu!(nm,tpars)
    if isa(jac,Diagonal)                # can update tgrad in place
        A_mul_B!(jac,tgrad(nm))
    else
        updttg = jac * (tg = tgrad(nm))
        tg[:] = updttg
    end
    res
end

type CompositePLModF{T<:FP} <: PLregModF{T}
    pt::Ptrans
    plm::PLregModF{T}
end

function (*)(vv::Vector{DataType},plm::NLregModF)
    dt = Dtrans(vv)
    length(dt) == size(plm,1) || error("DimensionMisMatch")
    CompositePLModF(dt,plm)
end

function initpars(cp::CompositePLModF)
    ini = initpars(plmod(cp))
    pt = ptran(cp)
    res = similar(ini) # a comprehension might be better but the result has type Vector{Any}
    for i in 1:length(ini)
        res[i] = invtrans(pt[i],ini[i])
    end
    res
end

mmjac(cp::CompositePLModF) = mmjac(plmod(cp))

model_response(cp::CompositePLModF) = model_response(plmod(cp))

npars(cp::CompositePLModF) = npars(cp.plm)

plmod(cp::CompositePLModF) = cp.plm

function pnames(cp::CompositePLModF)
    plm = plmod(cp)
    nnl,nl,_ = size(plm)
    innms = pnames(plm)
    [innms[1:nl],pnames(ptran(cp),innms[nl + (1:nnl)])]
end

ptran(cp::CompositePLModF) = cp.pt

residuals(cp::CompositePLModF) = residuals(cp.plm)

size(cp::CompositePLModF) = size(plmod(cp))

size(cp::CompositePLModF,args...) = size(plmod(cp),args...)

tgrad(cp::CompositePLModF) = tgrad(cp.plm)

function updtMM!(cp::CompositePLModF,nlpars::Vector)
    plm = plmod(cp)
    nnl,nl,n = size(plm)
    length(nlpars) == nnl || error("DimensionMismatch")
    tpars, jac = parjac(ptran(cp), nlpars)
    res = updtMM!(plm,tpars)
    if isa(jac,Diagonal)                # can update mmjac in place
        A_mul_B!(jac,reshape(mmjac(plm),(nnl,nl*n)))
    else
        mm = mmjac(plm)
        updtmmd = jac * reshape(mm,(nnl,nl*n))
        mm[:] = updtmmd
    end
    res
end

## There should be a version of this that does not need to update the nlpars part.
## See the definition of the deviance method for PLinearLS, Vector in plreg.jl
function updtmu!(cp::CompositePLModF,pars::Vector)
    plm = plmod(cp)
    nnl,nl,n = size(plm)
    length(pars) == nnl + nl || error("DimensionMisMatch")
    tpars, jac = parjac(ptran(cp),pars[nl + (1:nnl)])
    res = updtmu!(plm,[pars[1:nl],tpars])
    if isa(jac,Diagonal)
        A_mul_B!(jac,reshape(mmjac(plm),(nnl,nl*n)))
        scale!([ones(nl),diag(jac)],tgrad(plm))
    else
        error("code not yet written")
    end
    res
end
