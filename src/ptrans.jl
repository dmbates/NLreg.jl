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

Base.getindex(d::Dtrans,i) = d.vv[i]

Base.length(d::Dtrans) = length(d.vv)

function parjac(d::Dtrans, inpar::Vector)
    (p = length(d)) == length(inpar) || throw(DimensionMismatch(""))
    outpar = similar(inpar)
    jdiag = similar(inpar)
    for i in 1:p
        outpar[i], jdiag[i] = parjac(d[i],inpar[i])
    end
    (outpar, Diagonal(jdiag))                              
end

function pnames(d::Dtrans, innms::Vector)
    (p = length(d)) == length(innms) || throw(DimensionMismatch(""))
    [pnames(d[i],innms[i]) for i in 1:p]
end

## The scalar transformations are named according to the mapping from the
## model's parameters to the composite model's parameters, called invtrans here.
## This is in the grand tradition of GLMs where the important transformation is called
## the "inverse link".
type LogTr <: Strans end                # logarithmic transformation
parjac(::Type{LogTr}, x) = (ex = exp(x); (ex, ex))
pnames(::Type{LogTr}, nm) = "log(" * nm * ")"
invtrans(::Type{LogTr}, x) = log(x)

type IdTr <: Strans end                 # identity transformation
parjac(::Type{IdTr}, x) = (x,one(x))
pnames(::Type{IdTr}, nm) = nm
invtrans(::Type{IdTr}, x) = x


