using Base.Test
using DataFrames, NLreg

const sd1 = within(readtable(Pkg.dir("NLreg","data","sd1.csv.gz")),:(ID = pool(ID)))

nl = NonlinearLS(logsd1(:(CONC ~ TIME), sd1))

@test_approx_eq coef(nl) [0.13355908560885243,-1.4038489231938387]
