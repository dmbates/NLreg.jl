using Base.Test
using DataFrames, NLreg

const sd1 = within(readtable(Pkg.dir("NLreg","data","sd1.csv.gz")),:(ID = pool(ID)))

nl = gnfit(NonlinearLS(Logsd1(:(CONC ~ TIME), sd1)),true)

pl = gpfit(PLinearLS(LogBolusSD1(:(CONC ~ TIME), sd1)),true)

@test_approx_eq coef(nl) [0.1336302830419119,-1.4037023962519823]

pl = fit(AsympReg([2.:7],[18.6,22.6,25.1,27.2,29.1,30.1]),true)
