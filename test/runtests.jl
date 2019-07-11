using CSV, DataFrames, NLreg, StatsBase, Test

function sdOral1C(φ, data)
    k  = exp(φ[1])    # elimination rate constant from lk
    ka = exp(φ[2])    # absorption rate constant from lka
    V  = exp(φ[3])    # volume of distribution from lV
    t  = data.time
    @. (data.dose/V) * (ka/(ka - k)) * (exp(-k*t) - exp(-ka*t))
end

const Theo = CSV.read(joinpath(dirname(pathof(NLreg)),"..", "data","Theophylline.csv"));

m1 = fit!(NLregModel(sdOral1C, Theo, (lk = -2.5, lka = 0.5, lV = -1.0), :conc));

@test isapprox(m1.φ, [-2.5242397696363748, 0.39922827298046404, -0.7240233688477656], atol=1.0e-6)
@test isapprox(rss(m1), 274.4491345682938, atol=1.0e-6)

# The Puromycin example from Bates and Watts (1988)
data = DataFrame(conc = repeat([0.02,0.06,0.11,0.22,0.56,1.1], inner=2))
m2 = fit!(NLregModel((x,data)->@.(x[1]*data.conc/(x[2] + data.conc)), data,
    float([76,47,97,107,123,139,159,152,191,201,207,200]), (Vm=200.,K=0.05)))
@test isapprox(m2.φ, [212.68389364030205, 0.0641215161457169], rtol=1.e-5)
@test isapprox(stderror(m2), [6.947163392027966, 0.008280974562694107], rtol=1.e-5)
@test isapprox(deviance(m2), 1195.4488145259597, rtol=1.e-5)

#=
@test_approx_eq coef(pl) [1.1429558452268844,-1.4036989045425874]

pl = fit(AsympReg([2.:7],[18.6,22.6,25.1,27.2,29.1,30.1]),true)

const bod = DataFrame(Time=[1.:5,7],Demand=[8.3,10.3,19.0,16.0,15.6,19.8])

pl = fit(AsympOrig(Demand ~ Time,bod),true)
=#