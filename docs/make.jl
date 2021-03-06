using Documenter, NLreg, StatsBase

makedocs(;
    modules=[NLreg, StatsBase],
    format=Documenter.HTML(assets=String[]),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/dmbates/NLreg.jl/blob/{commit}{path}#L{line}",
    sitename="NLreg.jl",
    authors="Douglas Bates <dmbates@gmail.com>",
)

deploydocs(;
    repo="github.com/dmbates/NLreg.jl",
)
