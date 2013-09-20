immutable MicMen{T<:FP} <: PLregMod{T}
    x::Vector{T}
end
MicMen{T<:FP}(c::DataArray{T,1}) = MicMen(vector(c))
MicMen{T<:Integer}(c::DataArray{T,1}) = MicMen(float(vector(c)))
pnames(m::MicMen) = ["Vm", "K"]
size(m::MicMen) = (length(m.x),1,1)

function updtMM!{T<:FP}(m::MicMen{T},K::T,MM::Matrix{T},MMD::Array{T,3})
    x = m.x
    for i in 1:length(x)
        xi = x[i]
        denom = K + xi
        MMD[i,1,1] = -(MM[i,1] =  xi/denom)/denom
    end
    MM
end
updtMM!{T<:FP}(m::MicMen{T},nlp::Vector{T},MM::Matrix{T},MMD::Array{T,3}) = updtMM!(m,nlp[1],MM,MMD)
function updtMM!{T<:FP}(m::MicMen{T},nlp::Matrix{T},MM::Matrix{T},MMD::Array{T,3})
    n, nl, nnl = size(m); size(nlp) == (n,nnl) || error("Dimension mismatch")
    x = m.x
    for i in 1:length(x)
        xi = x[i]; Ki = nlp[i,1]
        denom = Ki + xi
        MMD[i,1,1] = -(MM[i,1] =  xi/denom)/denom
    end
    MM
end

immutable AsympReg{T<:FP} <: PLregMod{T}
    x::Vector{T}
end
AsympReg{T<:FP}(c::DataArray{T,1}) = AsympReg(vector(c))
AsympReg{T<:Integer}(c::DataArray{T,1}) = AsympReg(float(vector(c)))

pnames(m::AsympReg) = ["Asym","R0","rc"]

function updtMM!{T<:FP}(m::AsympReg{T},rc::T,MM::Matrix{T},MMD::Array{T,3})
    x = m.x
    for i in 1:length(x)
        xi = x[i]
        MMD[i,2,1] = -xi*(MM[i,2] = exp(-rc*xi))
    end
    MM
end
updtMM!{T<:FP}(m::AsympReg{T},nlp::Vector{T},MM::Matrix{T},MMD::Array{T,3}) = updtMM!(m,nlp[1],MM,MMD)

immutable Exp1{T<:FP} <: PLregMod{T}
    t::Vector{T}
end
Exp1{T<:FP}(t::DataArray{T,1}) = Exp1(vector(t))
Exp1{T<:Integer}(t::DataArray{T,1}) = Exp1(float(vector(t)))
pnames(m::Exp1) = ["V","ke"]
size(m::Exp1) = (length(m.x),1,1)
updt1!{T<:FP}(m::Exp1,t::T,ke::T,mm::StridedVector{T},mmd::StridedMatrix{T}) = mmd[1,1] = -t*(mm[1] = exp(-ke*t))

function updtMM!{T<:Float64}(m::Exp1{T},ke::T,MM::Matrix{T},MMD::Array{T,3})
    t=m.t; for i in 1:length(t) updt1!(m,t[i],ke,sub(MM,:,i),sub(MMD,:,:,i)) end
    MM
end
function updtMM!{T<:Float64}(m::Exp1{T},nlp::Vector{T},MM::Matrix{T},MMD::Array{T,3})
    t=m.t; ke = nlp[1]
    for i in 1:length(t) updt1!(m,t[i],ke,sub(MM,:,i),sub(MMD,:,:,i)) end
    MM
end
function updtMM!{T<:Float64}(m::Exp1{T},nlp::Matrix{T},MM::Matrix{T},MMD::Array{T,3})
    t=m.t; for i in 1:length(t) updt1!(m,t[i],nlp[i,1],sub(MM,:,i),sub(MMD,:,:,i)) end
    MM
end
