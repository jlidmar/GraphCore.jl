using GraphCore
using Documenter

DocMeta.setdocmeta!(GraphCore, :DocTestSetup, :(using GraphCore); recursive=true)

makedocs(;
    modules=[GraphCore],
    authors="Jack Lidmar <jlidmar@kth.se> and contributors",
    sitename="GraphCore.jl",
    format=Documenter.HTML(;
        canonical="https://jlidmar@kth.se.github.io/GraphCore.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jlidmar@kth.se/GraphCore.jl",
    devbranch="main",
)
