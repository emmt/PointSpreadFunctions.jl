using Documenter

push!(LOAD_PATH, "../src/")
using PointSpreadFunctions

DEPLOYDOCS = (get(ENV, "CI", nothing) == "true")

makedocs(;
         modules=[PointSpreadFunctions],
         format=Documenter.HTML(;
                                prettyurls = DEPLOYDOCS,
                                ),
         pages=[
             "Home" => "index.md",
             "Reference" => "library.md",
         ],
         repo="https://github.com/emmt/PointSpreadFunctions.jl/blob/{commit}{path}#L{line}",
         sitename = "Point spread functions for Julia",
         authors="Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>",
)

if DEPLOYDOCS
    deploydocs(;
               repo="github.com/emmt/PointSpreadFunctions.jl",
               )
end
