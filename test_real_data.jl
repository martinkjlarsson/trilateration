using CSV, DataFrames, Statistics, Plots

include("trilat.jl")
include("trilat_previous_works.jl")


function trilat_real(wifis, scans, include_rtt, include_rss, use_weights)
    # Noise levels for RSS and RTT measurements.
    rss_std = 5.0
    rtt_std = 1.0

    # Calibrated offset for RTT measurements. No offset needed.
    rtt_offset = 0

    alls = zeros(2, 0)
    allx = zeros(2, 0)
    allxml = zeros(2, 0)
    allxgt = zeros(2, 0)
    errors = zeros(0)
    errorsml = zeros(0)

    bssid2index = Dict(wifis.bssid .=> 1:size(wifis, 1))

    gscans = groupby(scans, :scanId)
    gkeys = keys(gscans)

    # For each scan.
    for key in gkeys
        group = gscans[key]

        s = zeros(2, 0)
        d2 = zeros(0)
        w = zeros(0)

        rtt_s = zeros(2, 0)
        rtt_d = zeros(0)

        rss_s = zeros(2, 0)
        rss_C = zeros(0)
        rss_C0 = zeros(0)
        rss_eta = zeros(0)

        # For each measurement in scan.
        for srow in eachrow(group)
            wifi = wifis[bssid2index[srow.bssid], :]
            si = [wifi.x, wifi.y]

            # Add RTT measurment.
            if include_rtt
                d = srow.rttDist / 1000 - rtt_offset
                d2i = d^2
                wi = 1 / (4 * d2i)
                wi /= rtt_std^2

                rtt_s = [rtt_s si]
                push!(rtt_d, d)

                s = [s si]
                push!(d2, d2i)
                push!(w, wi)
            end

            # Add RSS measurment.
            if include_rss
                C = srow.rssi
                C0 = wifi.txPower
                η = wifi.pathLossExponent

                rss_s = [rss_s si]
                push!(rss_C, C)
                push!(rss_C0, C0)
                push!(rss_eta, η)

                d2i = 10^((C0 - C) / (5η))
                wi = (5 * η / (d2i * log(10)))^2
                wi /= rss_std^2

                s = [s si]
                push!(d2, d2i)
                push!(w, wi)
            end
        end

        if length(d2) < 2
            @warn "Too few measurements. Skipping scan point."
            continue
        end

        # Trilaterate.
        println("Trilateration with $(length(d2)) measurements")
        W = use_weights ? Diagonal(w) : I(length(w))
        x = trilat(s, d2, W)

        if size(x, 2) > 1
            @error "Multiple solutions"
        end
        x = vec(x[:, 1])

        xgt = [first(group.x), first(group.y)]
        xml = trilat_ml_rtt_rss(
            rtt_s,
            rtt_d,
            rtt_std,
            rss_s,
            rss_C,
            rss_C0,
            rss_eta,
            rss_std,
            x0 = xgt,
        )

        alls = [alls s]
        allx = [allx x]
        allxml = [allxml xml]
        allxgt = [allxgt xgt]

        push!(errors, norm(x - xgt))
        push!(errorsml, norm(xml - xgt))
    end

    return errors, errorsml, alls, allx, allxml, allxgt
end


# Load data.
wifis = CSV.read("data/wifis.csv", DataFrame)
scans = CSV.read("data/scans.csv", DataFrame)

# Change depending on measurement types and weighting.
include_rtt = true
include_rss = false
use_weights = true

# Trilaterate.
errors, errorsml, s, x, xml, xgt =
    trilat_real(wifis, scans, include_rtt, include_rss, use_weights)
s = unique(s, dims = 2)

# Print results.
println()
println("Receivers:     $(length(errors))")
println("Senders:       $(size(s,2))")
println("Median error:  $(median(errors))")
println("RMS error:     $(sqrt(mean(errors.^2)))")
println("Mean error:    $(mean(errors))")
println("Mean error ML: $(mean(errorsml))")

# Plot results.
p = plot(
    aspect_ratio = :equal,
    grid = :none,
    xlim = (-20, 16),
    ylim = (-12, 11),
    xticks = -20:5:15,
    size = (500, 350),
    xlabel = "x/m",
    ylabel = "y/m",
    legend = :bottomright,
)
plot!([xgt[1, :]'; x[1, :]'], [xgt[2, :]'; x[2, :]'], color = :gray, label = "")
plot!([xgt[1, :]'; xml[1, :]'], [xgt[2, :]'; xml[2, :]'], color = :gray, label = "")
scatter!(s[1, :], s[2, :], label = "sⱼ", color = :lime)
scatter!(xgt[1, :], xgt[2, :], label = "x GT", color = :black, markerstrokecolor = :black)
scatter!(xml[1, :], xml[2, :], label = "x ML", color = :blue)
scatter!(x[1, :], x[2, :], label = "x Proposed", color = :red)

display(p)
