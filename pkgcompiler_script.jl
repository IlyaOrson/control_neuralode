using PackageCompiler

@time create_sysimage(
    [
        :OhMyREPL,
        :Revise,
        :DiffEqFlux,
        :Flux,
        :Optim,
        :OrdinaryDiffEq,
        :DiffEqSensitivity,
        :Zygote,
        :GalacticOptim,
        :UnicodePlots,
        :ClearStacktrace,
        :Dates,
        :CSV,
        :Tables,
    ],
    sysimage_path="sys_diffflux.so",
    precompile_execution_file="src/_precompile_execution_file.j"
)
