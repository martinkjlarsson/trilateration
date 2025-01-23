using StatsPlots
using DataFrames
using ColorSchemes

include("trilat.jl")
include("trilat_previous_works.jl")

## Setup.
m = 6
dim = 3
iters = 100
# iters = 1000 # Used in the paper.
tol = 1e-16
success_tol = 1e-6

scale = 1; # Global scale of problem.
xscales = exp10.(-12:0.5:0) # Scale of sender x-coordinate.
sigma = 0.0


## Define solvers.
W = d2 -> Diagonal(1 ./ max.(d2, 1e-4))
# W = d2 -> Diagonal(1 ./ d2)
# W = d2 -> I(length(d2))

solvers = Vector{Tuple{String,Function}}()
push!(solvers, ("Zhou", (s, d, x) -> trilat_zhou(s, d .^ 2)))
# push!(solvers, ("Zhou Exhaustive", (s, d, x) -> trilat_zhou_exhaustive(s, d .^ 2)))
push!(solvers, ("Linear", (s, d, x) -> trilat_linear(s, d .^ 2, W(d .^ 2))))
push!(solvers, ("Beck SDR", (s, d, x) -> trilat_beck_sdr(s, d)))
push!(solvers, ("Ismailova", (s, d, x) -> trilat_ismailova(s, d, sigma = tol)))
push!(solvers, ("Adachi", (s, d, x) -> trilat_adachi(s, d .^ 2, W(d .^ 2))))
push!(solvers, ("Luke", (s, d, x) -> trilat_luke(s, d, tol = tol)))
push!(solvers, ("Beck SFP", (s, d, x) -> trilat_beck_sfp(s, d, tol = tol)))
push!(solvers, ("Beck SR-LS", (s, d, x) -> trilat_beck_srls(s, d .^ 2, W(d .^ 2), tol = 0)))
push!(solvers, ("Proposed (Alg. 1)", (s, d, x) -> trilat_A(s, d .^ 2, W(d .^ 2))))
push!(solvers, ("Proposed (Alg. 2)", (s, d, x) -> trilat(s, d .^ 2, W(d .^ 2))))
# push!(solvers, ("ML", (s, d, x) -> trilat_ml(s, d, x0 = x)))


## Test.
median_errors = fill(Inf, length(xscales), length(solvers))
success_rate = zeros(length(xscales), length(solvers))
errors = DataFrame(solver = Int[], scale = [], error = [])
for isolver in 1:length(solvers)
    solver = solvers[isolver]
    println("Performing tests for $(solver[1]).")
    for (iscale, xscale) in enumerate(xscales)
        errors_iter = fill(Inf, iters)
        for iter in 1:iters
            x = scale * randn(dim)
            s = scale * randn(dim, m)
            s[1, :] *= xscale # Note: Scaling the x-coordinate is bad for Zhou.
            d = [norm(x - s[:, j]) for j in 1:m]
            d += sigma * randn(m)

            x_est = solver[2](s, d, x)

            if isnothing(x_est)
                err = Inf
            else
                err = minimum(norm(x_esti - x) for x_esti in eachcol(x_est))
            end

            errors_iter[iter] = err
            push!(errors, [isolver xscale err])
        end
        median_errors[iscale, isolver] = median(errors_iter)
        success_rate[iscale, isolver] = count(errors_iter .< success_tol) / iters
    end
end

## Plot.
names = reshape([solvers[i][1] for i in 1:length(solvers)], 1, :)

gr()

psuccess_rate = plot(
    xscales,
    success_rate,
    lw = 3,
    legend = :outerright,
    label = names,
    xaxis = :log,
    palette = :tab10,
    gridalpha = 1.0,
    minorgrid = false,
    grid = true,
    foreground_color_grid = :lightgray,
    xticks = 10.0 .^ (-12:4:0),
    yticks = 0:0.2:1,
    xlabel = "Scaling factor",
    ylabel = "Success rate",
    ylims = (-0.02, 1.02),
    size = (500, 300),
)
display(psuccess_rate)

pmedian = plot(
    xscales,
    median_errors,
    lw = 3,
    legend = :outerright,
    label = names,
    xaxis = :log,
    yaxis = :log,
    palette = :tab10,
    gridalpha = 1.0,
    minorgrid = false,
    grid = true,
    foreground_color_grid = :lightgray,
    xticks = 10.0 .^ (-12:4:0),
    yticks = 10.0 .^ (-16:4:0),
    xlabel = "Scaling factor",
    ylabel = "Median error",
    ylims = (5e-17, 1e3),
    size = (500, 300),
)
display(pmedian)

# savefig(psuccess_rate, "figs/success_rate.pdf")
# savefig(pmedian, "figs/degenerate_no_noise.pdf")
