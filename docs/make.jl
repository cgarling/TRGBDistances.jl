using TRGBDistances
using Documenter
using DocumenterCitations: CitationBibliography

# Figure out if we're running in CI
ci = get(ENV, "CI", nothing) == "true"

DocMeta.setdocmeta!(TRGBDistances, :DocTestSetup, :(using TRGBDistances); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric) # style=:authoryear

makedocs(;
    modules=[TRGBDistances],
    authors="cgarling <chris.t.garling@gmail.com> and contributors",
    sitename="TRGBDistances.jl",
    format=Documenter.HTML(;
        prettyurls = ci,
        canonical="https://cgarling.github.io/TRGBDistances.jl",
        edit_link="main",
        assets=String["assets/citations.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Luminosity Function Modeling" => [
            "Theory" => "lf_modeling/theory.md",
        ],
        "References" => "refs.md",
    ],
    doctest=false,
    linkcheck=ci,
    warnonly=[:missing_docs, :linkcheck],
    plugins=[bib],
)

deploydocs(;
    repo="github.com/cgarling/TRGBDistances.jl",
    devbranch="main",
)
