using Documenter, PointSpreadFunctions

makedocs(;
    modules=[PointSpreadFunctions],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/emmt/PointSpreadFunctions.jl/blob/{commit}{path}#L{line}",
    sitename="PointSpreadFunctions.jl",
    authors="Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>",
    assets=String[],
)

deploydocs(;
    repo="github.com/emmt/PointSpreadFunctions.jl",
)
