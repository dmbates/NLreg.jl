using Base.Test
using DataFrames, NLreg

const sd1 = within(readtable(Pkg.dir("NLreg","data","sd1.csv.gz")),:(ID = pool(ID)))

nl = NonlinearLS(Logsd1(:(CONC ~ TIME), sd1))

@test_approx_eq coef(nl) [0.1336302830419119,-1.4037023962519823]
