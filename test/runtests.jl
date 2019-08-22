using CSV, DataFrames, NLreg, Test

const datadir = normpath(joinpath(dirname(pathof(NLreg)), "..", "data"))

@testset "orange" begin
    Orange = CSV.read(joinpath(datadir, "Orange.csv"))
    logist(p, d) = p.Asym/(1 + exp(-(d.age - p.xmid)/p.scal))
    m1 = fit(NLregModel, first(groupby(Orange, :tree)), :circumference, logist,
        (Asym = 200., xmid = 1000., scal = 500.))
    @test rss(m1) ≈ 176.9948616999825 rtol=1.0e-5
end

@testset "Theophylline" begin
    Theoph = CSV.read(joinpath(datadir, "Theophylline.csv"))
    function sdOral1Clog(p, d)
        k  = exp(p.lk)    # elimination rate constant from lk
        ka = exp(p.lka)    # absorption rate constant from lka
        V  = exp(p.lV)    # volume of distribution from lV
        t  = d.time
        (d.dose/V) * (ka/(ka - k)) * (exp(-k*t) - exp(-ka*t))
    end
    m2 = fit(NLregModel, first(groupby(Theoph,:Subj)), :conc, sdOral1Clog,
        (lk = -2.0, lka = 0.5, lV = -1.0))
    @test m2.current ≈ [-2.9196139311895775, 0.5751606356206513, -0.9962424923171767] rtol=1.e-5
    @test keys(coef(m2)) == (:lk, :lka, :lV)
    @test rss(m2) ≈ 4.286009024337077 rtol=1.e-5
end
#=
@test_approx_eq coef(pl) [1.1429558452268844,-1.4036989045425874]

pl = fit(AsympReg([2.:7],[18.6,22.6,25.1,27.2,29.1,30.1]),true)

const bod = DataFrame(Time=[1.:5,7],Demand=[8.3,10.3,19.0,16.0,15.6,19.8])

pl = fit(AsympOrig(Demand ~ Time,bod),true)
=#
