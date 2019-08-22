using Documenter, NLreg

makedocs(;
    modules=[NLreg],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/dmbates/NLreg.jl/blob/{commit}{path}#L{line}",
    sitename="NLreg.jl",
    authors="Douglas Bates <dmbates@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/dmbates/NLreg.jl",
)
