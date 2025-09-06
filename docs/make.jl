using GraphCore
using Documenter

DocMeta.setdocmeta!(GraphCore, :DocTestSetup, :(using GraphCore); recursive=true)

makedocs(;
    modules=[GraphCore],
    authors="Jack Lidmar <jlidmar@kth.se>",
    sitename="GraphCore.jl",
    format=Documenter.HTML(;
        canonical="https://jlidmar.github.io/GraphCore.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Index" => "docindex.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/jlidmar/GraphCore.jl",
    devbranch="main",
)
