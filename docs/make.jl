using Phunny
using Documenter

DocMeta.setdocmeta!(Phunny, :DocTestSetup, :(using Phunny); recursive=true)

makedocs(;
    modules=[Phunny],
    authors="Isaac Ownby, Immanuel Schmidt",
    sitename="Phunny.jl",
    format=Documenter.HTML(;
        canonical="https://mani149.github.io/Phunny.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mani149/Phunny.jl",
    devbranch="main",
)
