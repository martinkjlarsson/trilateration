using StatsPlots
using Plots.PlotMeasures
using DataFrames
using ColorSchemes

include("trilat.jl")
include("trilat_previous_works.jl")

## Setup.
m = 10
dim = 3
iters = 1000
# iters = 10000 # Used in paper. Takes ~30 min to run.
sigmas = [0.001, 0.01, 0.1]

## Define solvers.
W = d2 -> Diagonal(1 ./ d2)
# W = d2 -> I(length(d2))

solvers = Vector{Tuple{String,Function}}()
push!(solvers, ("Zhou", (s, d, x) -> trilat_zhou(s, d .^ 2)))
# push!(solvers, ("Zhou Exhaustive", (s, d, x) -> trilat_zhou_exhaustive(s, d .^ 2)))
push!(solvers, ("Linear", (s, d, x) -> trilat_linear(s, d .^ 2, W(d .^ 2))))
push!(solvers, ("Beck SDR", (s, d, x) -> trilat_beck_sdr(s, d)))
push!(solvers, ("Ismailova", (s, d, x) -> trilat_ismailova(s, d)))
push!(solvers, ("Adachi", (s, d, x) -> trilat_adachi(s, d .^ 2, W(d .^ 2))))
push!(solvers, ("Luke", (s, d, x) -> trilat_luke(s, d)))
push!(solvers, ("Beck SFP", (s, d, x) -> trilat_beck_sfp(s, d)))
push!(solvers, ("Beck SR-LS", (s, d, x) -> trilat_beck_srls(s, d .^ 2, W(d .^ 2))))
push!(solvers, ("Proposed (Alg. 1)", (s, d, x) -> trilat_A(s, d .^ 2, W(d .^ 2))))
push!(solvers, ("Proposed (Alg. 2)", (s, d, x) -> trilat(s, d .^ 2, W(d .^ 2))))
# push!(solvers, ("Proposed (Alg. 2, W=I)", (s, d, x) -> trilat(s, d .^ 2, I(m))))
push!(solvers, ("ML", (s, d, x) -> trilat_ml(s, d, x0 = x)))


## Test.
mean_errors = fill(Inf, length(sigmas), length(solvers))
median_errors = fill(Inf, length(sigmas), length(solvers))
errors = DataFrame(solver = Int[], sigma = [], error = [])
for isolver in 1:length(solvers)
    solver = solvers[isolver]
    println("Performing tests for $(solver[1])")
    for isigma in eachindex(sigmas)
        sigma = sigmas[isigma]

        errors_iter = fill(Inf, iters)
        for iter in 1:iters
            x = randn(dim)
            s = randn(dim, m)
            d = [norm(x - s[:, j]) for j in 1:m]
            d += sigma * randn(m)

            x_est = solver[2](s, d, x)

            if !isnothing(x_est)
                err = minimum(norm(x_est[:, i] - x) for i in axes(x_est, 2))
                errors_iter[iter] = err
                push!(errors, [isolver sigma err])
            else
                @error "Solver failed! Error is not saved!"
            end
        end
        mean_errors[isigma, isolver] = mean(errors_iter)
        median_errors[isigma, isolver] = median(errors_iter)
    end
end

## Plot.
names = reshape([solvers[i][1] for i in 1:length(solvers)], 1, :)
cs = [colorschemes[:tab10].colors; RGB(0.8, 0.8, 0.8)]

gr()

p = groupedboxplot(
    errors[:, 2],
    errors[:, 3] ./ errors[:, 2],
    group = errors[:, 1],
    outliers = false,
    legend = :outerbottomright,
    palette = cs,
    grid = :y,
    label = names,
    xlabel = "Noise σ",
    ylabel = "Normalized error (error/σ)",
    size = (1000, 300),
    left_margin = 5mm,
    bottom_margin = 5mm,
)
display(p)

# savefig(p, "figs/numerics_wide.pdf")
