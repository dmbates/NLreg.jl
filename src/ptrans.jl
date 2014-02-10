abstract Ptrans
abstract Strans                         # scalar parameter transformation

type Dtrans <: Ptrans                   # diagonal (i.e. vector of scalars)
    vv::Vector{DataType}
    function Dtrans(vv::Vector{DataType})
        for d in vv
            d <: Strans || error("type $d is not an Strans")
        end
        new(vv)
    end
end

getindex(d::Dtrans,i) = d.vv[i]

length(d::Dtrans) = length(d.vv)

function parjac(d::Dtrans, inpar::Vector)
    (p = length(d)) == length(inpar) || error("DimensionMismatch")
    outpar = similar(inpar)
    jdiag = similar(inpar)
    for i in 1:p
        outpar[i], jdiag[i] = parjac(d[i],inpar[i])
    end
    (outpar, Diagonal(jdiag))                              
end

function pnames(d::Dtrans, innms::Vector)
    (p = length(d)) == length(innms) || error("DimensionMismatch")
    [pnames(d[i],innms[i]) for i in 1:p]
end
    
type ExpTr <: Strans end                # exponential transformation
parjac(::Type{ExpTr}, x) = (ex = exp(x); (ex, ex))
pnames(::Type{ExpTr}, nm) = "log(" * nm * ")"
invtrans(::Type{ExpTr}, x) = log(x)

type IdTr <: Strans end                 # identity transformation
parjac(::Type{IdTr}, x) = (x,one(x))
pnames(::Type{IdTr}, nm) = nm
invtrans(::Type{IdTr}, x) = x


