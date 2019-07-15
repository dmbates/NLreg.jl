@testset "TheoNLreg" begin
    m1 = fit!(NLregModel(sdOral1C, Theo, (lk = -2.5, lka = 0.5, lV = -1.0), :conc));

    @test isapprox(m1.φ, [-2.5242397696363748, 0.39922827298046404, -0.7240233688477656], atol=1.0e-6)
    @test isapprox(rss(m1), 274.4491345682938, atol=1.0e-6)
end

# The Puromycin example from Bates and Watts (1988)
@testset "PuromycinTreated" begin
    data = DataFrame(conc = repeat([0.02,0.06,0.11,0.22,0.56,1.1], inner=2))
    m2 = fit!(NLregModel((x,data)->@.(x[1]*data.conc/(x[2] + data.conc)), data,
        float([76,47,97,107,123,139,159,152,191,201,207,200]), (Vm=200.,K=0.05)))
    @test isapprox(m2.φ, [212.68389364030205, 0.0641215161457169], rtol=1.e-5)
    @test isapprox(stderror(m2), [6.947163392027966, 0.008280974562694107], rtol=1.e-5)
    @test isapprox(deviance(m2), 1195.4488145259597, rtol=1.e-5)
end
