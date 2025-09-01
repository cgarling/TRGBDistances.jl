using TRGBDistances
using Documenter

DocMeta.setdocmeta!(TRGBDistances, :DocTestSetup, :(using TRGBDistances); recursive=true)

makedocs(;
    modules=[TRGBDistances],
    authors="cgarling <chris.t.garling@gmail.com> and contributors",
    sitename="TRGBDistances.jl",
    format=Documenter.HTML(;
        canonical="https://cgarling.github.io/TRGBDistances.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    doctest=false,
    linkcheck=true,
    warnonly=[:missing_docs, :linkcheck],
)

deploydocs(;
    repo="github.com/cgarling/TRGBDistances.jl",
    devbranch="main",
)
