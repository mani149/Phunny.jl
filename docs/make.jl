using Phunny
using Documenter

DocMeta.setdocmeta!(Phunny, :DocTestSetup, :(using Phunny); recursive=true)

makedocs(;
    modules=[Phunny],
    authors="Isaac Ownby, Immanuel Schmidt",
    sitename="Phunny.jl",
    format=Documenter.HTML(;
    	prettyurls = get(ENV, "CI", "false") == "true",
        canonical="https://mani149.github.io/Phunny.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => "tutorials.md",
        "References" => "refs.md"
    ],
)

deploydocs(;
    repo="github.com/mani149/Phunny.jl.git",
    devbranch="main",
)
