using CSV, DataFrames, NLreg, Test

function sdOral1C(φ, data)
    k  = exp(φ[1])    # elimination rate constant from lk
    ka = exp(φ[2])    # absorption rate constant from lka
    V  = exp(φ[3])    # volume of distribution from lV
    t  = data.time
    @. (data.dose/V) * (ka/(ka - k)) * (exp(-k*t) - exp(-ka*t))
end

const Theo = CSV.read(joinpath(dirname(pathof(NLreg)),"..", "data","Theophylline.csv"));

m1 = fit!(NLregModel(sdOral1C, Theo, copy(Theo.conc), (lk = -2.5, lka = 0.5, lCl = -1.0)))

params(m1)

#=
@test_approx_eq coef(pl) [1.1429558452268844,-1.4036989045425874]

pl = fit(AsympReg([2.:7],[18.6,22.6,25.1,27.2,29.1,30.1]),true)

const bod = DataFrame(Time=[1.:5,7],Demand=[8.3,10.3,19.0,16.0,15.6,19.8])

pl = fit(AsympOrig(Demand ~ Time,bod),true)
=#