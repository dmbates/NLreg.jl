using Base.Test
using DataFrames, NLreg

sd1 = readtable(Pkg.dir("NLreg","data","sd1.csv.gz"))
sd1[:ID] = pool(sd1[:ID])

# nl = gnfit(NonlinearLS(Logsd1(:(CONC ~ TIME), sd1)),true)

pl = fit([LogTr] * BolusSD1(CONC ~ TIME, sd1),true)

@test_approx_eq coef(pl) [1.1429558452268844,-1.4036989045425874]

pl = fit(AsympReg([2.:7],[18.6,22.6,25.1,27.2,29.1,30.1]),true)
