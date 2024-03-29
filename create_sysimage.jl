# run without startup file
# julia --startup-file=no create_sysimage.jl

using Pkg
using PackageCompiler: create_sysimage

Pkg.activate(".")
deps = [pair.second for pair in Pkg.dependencies()]
direct_deps = filter(p -> p.is_direct_dep, deps)
pkg_name_version = [(x.name, x.version) for x in direct_deps]
pkg_list = [Symbol(x.name) for x in direct_deps]

create_sysimage(
    # pkg_list;
    ["OrdinaryDiffEq", "Enzyme", "ReverseDiff", "PyPlot", "Zygote", "SciMLSensitivity", "UnicodePlots", "DiffEqFlux", "InfiniteOpt"];
    sysimage_path="cnode.so",
    precompile_execution_file="sysimage_tracing.jl",
)

# how to use sysimage to execute examples & activate current project:
# julia --sysimage cnode.so --project=.
