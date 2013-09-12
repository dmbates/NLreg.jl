immutable MicMen{T<:FP} <: PLregMod{T}
    x::Vector{T}
    MM::Matrix{T}
    MMD::Array{T,3}
end
MicMen{T<:FP}(x::Vector{T}) = (n = length(x); MicMen(x, Array(T,n,1), Array(T,n,1,1)))
MicMen{T<:FP}(c::DataArray{T,1}) = MicMen(vector(c))
MicMen{T<:Integer}(c::DataArray{T,1}) = MicMen(convert(Vector{T},vector(c)))

pnames(m::MicMen) = ["Vm", "K"]

function newpar{T<:FP}(m::MicMen{T},K::T)
    x = m.x; MM = m.MM; MMD = m.MMD
@inbounds for i in 1:length(x)
        xi = x[i]
        denom = K + xi
        MMD[i,1,1] = -(MM[i,1] =  xi/denom)/denom
    end
    MM
end
function newpar{T<:FP}(m::MicMen{T},nlp::Vector{T})
    length(nlp) == 1 ? newpar(m,nlp[1]) : error("length(nlp) should be 1")
end

immutable AsympReg{T<:FP} <: PLregMod{T}
    x::Vector{T}
    MM::Matrix{T}
    MMD::Array{T,3}
end
AsympReg{T<:FP}(x::Vector{T}) = (n = length(x); AsympReg(x, ones(T,n,2), zeros(T,n,2,1)))
AsympReg{T<:FP}(c::DataArray{T,1}) = AsympReg(vector(c))
AsympReg{T<:Integer}(c::DataArray{T,1}) = AsympReg(float(vector(c)))

pnames(m::AsympReg) = ["Asym","R0","rc"]

function newpar{T<:FP}(m::AsympReg{T},rc::T)
    x = m.x; MM = m.MM; MMD = m.MMD
@inbounds for i in 1:length(x)
        xi = x[i]
        MMD[i,2,1] = -xi*(MM[i,2] = exp(-rc*xi))
    end
    MM
end
function newpar{T<:FP}(m::AsympReg{T},nlp::Vector{T})
    length(nlp) == 1 ? newpar(m,nlp[1]) : error("length(nlp) should be 1")
end
