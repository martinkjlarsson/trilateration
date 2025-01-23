using BenchmarkTools
using Random
using DataStructures
using Plots
using Printf

include("trilat.jl")
include("trilat_previous_works.jl")

function create_benchmark_suite(m, dim)
    x = zeros(dim)
    s = zeros(dim, m)
    d = zeros(m)
    d2 = zeros(m)

    function setup!(x, s, d, d2)
        randn!(x)
        randn!(s)
        for j in 1:m
            d[j] = norm(x - s[:, j])
            d2[j] = d[j]^2
        end
    end

    suite = BenchmarkGroup()
    suite["linear"] =
        @benchmarkable trilat_linear($s, $d2) setup = ($setup!($x, $s, $d, $d2))
    suite["zhou"] = @benchmarkable trilat_zhou($s, $d2) setup = ($setup!($x, $s, $d, $d2))
    suite["beck_sdr"] =
        @benchmarkable trilat_beck_sdr($s, $d) setup = ($setup!($x, $s, $d, $d2))
    suite["ismailova"] =
        @benchmarkable trilat_ismailova($s, $d) setup = ($setup!($x, $s, $d, $d2))
    suite["adachi"] =
        @benchmarkable trilat_adachi($s, $d2) setup = ($setup!($x, $s, $d, $d2))
    suite["luke"] = @benchmarkable trilat_luke($s, $d) setup = ($setup!($x, $s, $d, $d2))
    suite["beck_sfp"] =
        @benchmarkable trilat_beck_sfp($s, $d) setup = ($setup!($x, $s, $d, $d2))
    suite["beck_srls"] =
        @benchmarkable trilat_beck_srls($s, $d2) setup = ($setup!($x, $s, $d, $d2))
    suite["proposed_A"] =
        @benchmarkable trilat_A($s, $d2) setup = ($setup!($x, $s, $d, $d2))
    suite["proposed"] = @benchmarkable trilat($s, $d2) setup = ($setup!($x, $s, $d, $d2))

    return suite
end

function benchmark_all(m = 10, dim = 3)
    suite = create_benchmark_suite(m, dim)

    # Warm up for compilation purposes.
    BenchmarkTools.run(suite, evals = 1, samples = 1)

    # Actual benchmark.
    Random.seed!(0)
    results =
        BenchmarkTools.run(suite, verbose = true, evals = 1, seconds = 10, samples = 10000)

    return results
end

function create_table()
    ms = [4, 10, 100]

    results = [benchmark_all(m) for m in ms]

    names = OrderedDict{String,String}()
    names["zhou"] = "Zhou"
    names["linear"] = "Linear"
    names["beck_sdr"] = "Beck SDR"
    names["ismailova"] = "Ismailova"
    names["adachi"] = "Adachi"
    names["luke"] = "Luke"
    names["beck_sfp"] = "Beck SFP"
    names["beck_srls"] = "Beck SR-LS"
    names["proposed_A"] = "Proposed (Alg. 1)"
    names["proposed"] = "Proposed (Alg. 2)"

    for (id, name) in names
        if !haskey(results[1], id)
            println("Missing result for solver \"$name\"")
            continue
        end
        print(name)
        for i in 1:length(ms)
            microseconds = median(results[i][id]).time / 1000
            if microseconds >= 1000000
                @printf " & \\SI{%.1f}{\\second}" microseconds / 1000000
            elseif microseconds >= 1000
                @printf " & \\SI{%.1f}{\\milli\\second}" microseconds / 1000
            else
                @printf " & \\SI{%.1f}{\\micro\\second}" microseconds
            end
        end
        println(" \\\\")
    end
    return results
end

# Create table in paper.
results = create_table()
