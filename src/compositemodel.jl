type CompositeModF{T<:FP} <: NLregModF{T}
    pt::Ptrans
    nm::NLregModF{T}
    function CompositeModF(pt::Ptrans,nm::NLregMod)
        length(pt) == npars(nm) || error("DimensionMismatch")
        new(pt,nm)
    end
end

npars(cm::CompositeModF) = length(cm.pt)

function updtmu!(cm::CompositeModel,pars::Vector)
    length(pars) == npars(cm) || error("DimensionMismatch")
    tpars, jac = parjac(cm.pt, pars)
    res = updtmu!(n,tpars))
    m.tgrad[:] = jac * m.tgrad
end

function
