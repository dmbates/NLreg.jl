@testset "Orange" begin
    logisgrowth(p,d) = @.(p[1]/(1+exp(-(d.age - p[2])/p[3])))

    m = fit!(NLregModel(logisgrowth, Orange, (Asym=200.,xmid=700.,scal=350.),:circumference))
    nlmm = NLmixedModel(logisgrowth, groupby(Orange, :tree), :circumference, params(m))
    @test nlmm.θ == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    @test nlmm.φ == collect(params(m))
end
