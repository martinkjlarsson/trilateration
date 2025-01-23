using LinearAlgebra
using Statistics
using JuMP
using Hypatia
using Printf
using LsqFit
using Clarabel

include("qcqp.jl")

# Minimizes ∑ⱼwⱼ(α - 2xᵀsⱼ + sⱼᵀsⱼ - dⱼ²)² over (x, α) and returns x.
function trilat_linear(s, d2, W = Diagonal(1 ./ d2))
    A = [-2 * s' ones(size(s, 2))]
    b = d2 - vec(sum(abs2, s, dims = 1))
    y = (A' * W * A) \ (A' * W * b)
    return y[1:end-1]
end

# Y. Zhou, “A closed-form algorithm for the least-squares trilateration problem,” Robotica, vol. 29, no. 3, pp. 375–389, May 2011, doi: 10.1017/S0263574710000196.
function trilat_zhou(s, d2) #, W = Diagonal(1 ./ d2))
    # Use notation from paper.
    (n, N) = size(s)
    p = s
    r2 = d2

    # Some convenient precalculations.
    p2 = vec(sum(abs2, p, dims = 1))
    r2p2 = (sum(r2) - sum(p2)) / N
    pout = -2 / N * (p * p')

    # Calculate a, B, c.
    a = p * (p2 - r2) / N
    B = pout + r2p2 * I
    c = mean(p, dims = 2)
    c2 = dot(c, c)

    # Calculate a, B, c with weighting. Implementation not complete.
    # wsum = sum(W)
    # ws2md2 = W * (sum(abs2, p, dims = 1)' / wsum - r2)
    # a = vec(p * ws2md2)
    # B = -2 * p * W * p' / wsum - sum(ws2md2) * I
    # c = s * sum(W, dims = 2)
    # a ./= N
    # B ./= N
    # c ./= N
    # c2 = dot(c, c)

    # Calculate H, f.
    f = a + B * c + 2 * c2 * c
    H = pout + 2 * (c * c')
    # H = -2 / N * p * W * p' / wsum + 2 * (c * c')

    # Calculate Hp, hp, fp.
    # NOTE: Hp may not have full rank!
    fp = -(f[1:end-1] .- f[end])
    Hp = H[1:end-1, 1:end-1] .- H[end, 1:end-1]'
    hp = -(H[1:end-1, end] .- H[end, end])

    # Calculate q'*q.
    qq = r2p2 + c2

    # Calculate determinants.
    detHp = det(Hp)
    Hfh = copy(Hp)
    detHf = zeros(n - 1)
    detHh = zeros(n - 1)
    for k in 1:n-1
        Hfh[:, k] = fp
        detHf[k] = det(Hfh)
        Hfh[:, k] = hp
        detHh[k] = det(Hfh)
        Hfh[:, k] = Hp[:, k]
    end

    # Calculate coefficients for a1*qn^2 + a2*qn + a3.
    a1 = detHp^2 + sum(abs2, detHh)
    a2 = 2 * dot(detHf, detHh)
    a3 = -detHp^2 * qq + sum(abs2, detHf)

    # Solve quadratic and find qn.
    q = zeros(n, 2)
    discriminant = a2^2 - 4 * a1 * a3
    if discriminant > 0
        tmp = -(a2 + sign(a2) * sqrt(discriminant)) / 2
        q[end, 1] = tmp / a1
        q[end, 2] = a3 / tmp
    else
        # Avoid complex solutions and return something hopefully plausible.
        q[end, :] .= -a2 / (2 * a1)
    end

    # Calculate qk.
    q[1:n-1, :] .= detHf ./ detHp .+ detHh ./ detHp .* q[end, :]'

    # Calculate p0 (undo translation).
    p0 = q .+ c

    # Calculate cost. Not the same as S(p0), but found by integrating equation (5) in the paper.
    cost = [
        -dot(a + B * c + 2 * c2 * c, q[:, i]) -
        0.5 * q[:, i]' * (B + 2 * c * c' + c2 * I) * q[:, i] +
        0.25 * dot(q[:, i], q[:, i])^2 for i in 1:2
    ]

    # Sort the solutions based on cost.
    if cost[1] > cost[2]
        reverse!(p0, dims = 2)
    end

    # NOTE: There are always two solutions returned.
    return p0
end

# Since trilat_zhou fails with certain degenerate cases involving the last coordinate,
# we permute the coordinates and return a larger set of solutions.
function trilat_zhou_exhaustive(s, d2)
    n = size(s, 1)
    x = zeros(n, 0)
    for i in 1:n
        perm = [1:i-1; i+1:n; i]
        si = s[perm, :]
        xi = trilat_zhou(si, d2)
        inv_perm = [1:i-1; n; i:n-1]
        x = [x xi[inv_perm, :]]
    end
    return x
end

# A. Beck, P. Stoica, and J. Li, “Exact and Approximate Solutions of Source Localization Problems,” IEEE Transactions on Signal Processing, vol. 56, no. 5, pp. 1770–1778, May 2008, doi: 10.1109/TSP.2007.909342.
function trilat_beck_srls(s, d2, W = Diagonal(1 ./ d2); tol = 1e-5, maxiter = 300)
    (n, m) = size(s)

    # Calculate A, b, D, f.
    A = [-2 * s' ones(m)]
    b = d2 - vec(sum(abs2, s, dims = 1))
    D = Diagonal([ones(n); 0])
    f = [zeros(n); -0.5]

    # Precalculations.
    AA = A' * W * A
    Ab = A' * W * b

    # Definte ϕ(λ), which is strictly decreasing on the interval considered.
    function ϕ(λ)
        y = (AA + λ * D) \ (Ab - λ * f)
        y' * D * y + 2 * f' * y
    end

    # Find lower bound on lambda.
    lambdas = eigvals(D, AA)
    lb = -1 / maximum(lambdas)
    if lb >= 0
        @error @sprintf "Invalid lower bound for bisection (lb = %.1e), i.e., (AᵀA + λD) is not PSD." lb
        return nothing
    end

    # The paper does not mention a specific upper bound.
    ub = -lb
    while ϕ(ub) > 0
        lb = ub
        ub *= 2
    end

    # Use bisection to find the Lagrange multiplier λ.
    if tol <= 0.0
        tol = max(eps(lb), eps(ub))
    end
    iter = 0
    while ub - lb > tol
        mb = (lb + ub) / 2
        if ϕ(mb) < 0
            ub = mb
        else
            lb = mb
        end

        iter += 1
        if iter >= maxiter
            @info "Bisection reached maximum number of iterations ($maxiter)."
            break
        end
    end

    # Find final y and x.
    λ = (lb + ub) / 2
    y = (AA + λ * D) \ (Ab - λ * f)
    x = y[1:end-1]
    return x
end


# A. Beck, P. Stoica, and J. Li, “Exact and Approximate Solutions of Source Localization Problems,” IEEE Transactions on Signal Processing, vol. 56, no. 5, pp. 1770–1778, May 2008, doi: 10.1109/TSP.2007.909342.
# S. Adachi and Y. Nakatsukasa, “Eigenvalue-based algorithm and analysis for nonconvex QCQP with one constraint,” Math. Program., vol. 173, no. 1, pp. 79–116, Jan. 2019, doi: 10.1007/s10107-017-1206-8.
function trilat_adachi(s, d2, W = Diagonal(1 ./ d2))
    (n, m) = size(s)

    # Calculate (A, b, D, f) from Beck et al.
    AA = [-2 * s' ones(m)]
    bb = d2 - vec(sum(abs2, s, dims = 1))
    DD = Diagonal([ones(n); 0])
    ff = [zeros(n); -0.5]

    # Calculate (A, a, B, b, beta) from Adachi and Nakatsukasa.
    A = AA' * W * AA
    a = -AA' * W * bb
    B = DD
    b = ff
    beta = 0.0

    # Solve QCQP problem.
    # (A + lambdahat * B) needs to be positive definite. Any nonnegative value should work.
    lambdahat = 10
    y = qcqp(A, a, B, b, beta, lambdahat)
    if isnothing(y)
        return nothing
    end
    return y[1:end-1, :]
end


# A. Beck, P. Stoica, and J. Li, “Exact and Approximate Solutions of Source Localization Problems,” IEEE Transactions on Signal Processing, vol. 56, no. 5, pp. 1770–1778, May 2008, doi: 10.1109/TSP.2007.909342.
function trilat_beck_sdr(s, d; maxiter = 1000)
    (n, m) = size(s)

    # Construct problem using JuMP.
    # See https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers for all supported SDP solvers.
    model = Model(optimizer_with_attributes(
        Hypatia.Optimizer,
        "iter_limit" => maxiter,
        # "tol_rel_opt" => tol,
        # "tol_abs_opt" => tol,
    )) # 70 ms
    set_silent(model)
    @variable(model, G[1:m+1, 1:m+1], PSD)
    @variable(model, X[1:n+1, 1:n+1], PSD)
    @objective(model, Min, sum(G[i, i] - 2 * d[i] * G[m+1, i] + d[i]^2 for i in 1:m))
    @constraint(model, G[m+1, m+1] == 1)
    @constraint(model, X[n+1, n+1] == 1)
    for i in 1:m
        Ci = [I(n) -s[:, i]; -s[:, i]' dot(s[:, i], s[:, i])]
        @constraint(model, G[i, i] == tr(Ci * X))
    end

    # Solve SDP problem.
    optimize!(model)

    # Disregard optimizer status and return what was found.
    return value.(X[1:n, n+1])

    # Extract solution if successful.
    # if termination_status(model) == MOI.OPTIMAL ||
    #    termination_status(model) == MOI.ALMOST_OPTIMAL
    #     return value.(X[1:n, n+1])
    # else
    #     @error "Trilateration failed."
    #     return nothing
    # end
end

# A. Beck, M. Teboulle, and Z. Chikishev, “Iterative Minimization Schemes for Solving the Single Source Localization Problem,” SIAM J. Optim., vol. 19, no. 3, pp. 1397–1416, Jan. 2008, doi: 10.1137/070698014.
function trilat_beck_sfp(s, d; maxiter = 1000, tol = 1e-7, x0 = nothing)
    (n, m) = size(s)

    # Initialization.
    if isnothing(x0)
        f = x -> sum((norm(x - s[:, j]) - d[j])^2 for j in 1:m)
        k = argmin(j -> f(s[:, j]), 1:m)
        sk = s[:, k]

        ∇gk =
            2 * sum((1 - d[j] / norm(sk - s[:, j])) * (sk - s[:, j]) for j in 1:m if j != k)
        v0 = norm(∇gk) < tol ? ones(n) : -∇gk

        t = 1
        fsk = f(sk)
        while f(sk + t * v0) > fsk && t > 1e-6
            t /= 2
        end
        x = sk + t * v0
    else
        x = x0
    end

    ms = mean(s, dims = 2)
    px = zeros(n, 1)

    # Iterative part of algorithm.
    for _ in 1:maxiter
        px = x
        x = ms .+ mean(d[j] * normalize(x - s[:, j]) for j in 1:m)

        if norm(x - px) / norm(x) < tol
            # @info "Change in position is too small (k=$k)."
            break
        end
    end
    return x
end


# D. R. Luke, S. Sabach, M. Teboulle, and K. Zatlawey, “A simple globally convergent algorithm for the nonsmooth nonconvex single source localization problem,” J Glob Optim, vol. 69, no. 4, pp. 889–909, Dec. 2017, doi: 10.1007/s10898-017-0545-6.
function trilat_luke(s, d; maxiter = 1000, tol = 1e-7, rho = 2.00001, x0 = nothing)
    (n, m) = size(s)

    # The initialization from the paper is essentially random.
    # px = randn(n)
    # v0 = randn(n, m)
    # w0 = randn(n, m)

    # x = mean(s + v0, dims = 2)
    # # p = (v0 .+ w0 ./ rho') ./ d' # Paper says this.
    # p = (v0 .- w0 ./ rho') ./ d' # But this is probably correct.
    # u = p ./ sqrt.(max.(1, sum(abs2, p, dims = 1)))

    # Perhaps a more reasonable initialization.
    if isnothing(x0)
        ms = mean(s, dims = 2)
        ss = std(s, dims = 2)
        px = ms .+ ss .* randn(n)
        x = ms .+ ss .* randn(n) ./ sqrt(m)
        p = ss .* randn(n, m) ./ d'
        u = p ./ sqrt.(max.(1, sum(abs2, p, dims = 1)))
    else
        # Converges in one iteration if x0 is the true minimum and there is no noise.
        px = x0
        x = x0
        p = (x0 .- s) ./ d'
        u = p ./ sqrt.(max.(1, sum(abs2, p, dims = 1)))
    end

    for _ in 1:maxiter
        nx = mean(s .+ d' .* u .+ (x .- px) ./ rho', dims = 2)
        # p = u .+ (s .- px) ./ (rho' .* d') # From paper but wrong.
        p = u .+ (2 .* x .- s .- px) ./ (rho' .* d') # Correct.
        u = p ./ sqrt.(max.(1, sum(abs2, p, dims = 1)))

        px = x
        x = nx

        if norm(x - px) / norm(x) < tol
            # @info "Change in position is too small (k=$k)."
            break
        end
    end
    return x
end


# Ismailova, Darya, and Wu-Sheng Lu. “Penalty Convex-Concave Procedure for Source Localization Problem.” In 2016 IEEE Canadian Conference on Electrical and Computer Engineering (CCECE), 1–4. Vancouver, BC, Canada: IEEE, 2016. https://doi.org/10.1109/CCECE.2016.7726815.
function trilat_ismailova(
    s,
    d;
    maxiter = 20,
    tol = 1e-6,
    x0 = mean(s, dims = 2),
    tau0 = 10.0,
    taumax = 1e4,
    mu = 20.0,
    gamma = 3.0,
    sigma = 1e-6,
)
    (n, m) = size(s)

    normtol = 1e-16
    sbar = mean(s, dims = 2)
    delta = gamma .* sigma .* ones(m)

    tau = tau0
    previousx = x0
    currentx = x0
    for k in 1:maxiter
        # Calculate v from (10).
        v = sbar
        for j in 1:m
            nxs = norm(currentx - s[:, j])
            if nxs >= normtol
                v += d[j] / nxs * (currentx - s[:, j])
            end
        end
        v /= m

        # Solve convex problem (16).
        # See https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers for all supported SDP solvers.
        model = Model(Clarabel.Optimizer) # 4 ms
        # model = Model(COSMO.Optimizer) # 6 ms
        # model = Model(SCS.Optimizer) # 9 ms
        # model = Model(ECOS.Optimizer) # 15 ms
        set_silent(model)
        @variable(model, x[k = 1:n], start = currentx[k])
        @variable(model, slower[j = 1:m] >= 0)
        @variable(model, supper[j = 1:m] >= 0)
        for j in 1:m
            nxs = norm(currentx - s[:, j])
            @constraint(
                model,
                [supper[j] + d[j] + delta[j]; x - s[:, j]] in SecondOrderCone()
            )
            # @constraint(
            #     model,
            #     -nxs - dot(currentx - s[:, j], x - currentx) / nxs + d[j] - delta[j] <=
            #     slower[j]
            # )
            # Multiply the constraint above with nxs. This avoids numerical issues when nxs is close to 0.
            @constraint(
                model,
                -nxs^2 - dot(currentx - s[:, j], x - currentx) + nxs * (d[j] - delta[j]) <=
                nxs * slower[j]
            )
        end
        @objective(
            model,
            Min,
            dot(x, x) - 2 * dot(x, v) + tau * (sum(slower) + sum(supper))
        )

        optimize!(model)

        # Only continue if optimization is optimal or near-optimal.
        # if termination_status(model) != MOI.OPTIMAL &&
        #    termination_status(model) != MOI.ALMOST_OPTIMAL
        #     @error "Trilateration failed. No near-optimal iterate found."
        #     return nothing
        # end
        # if k == maxiter && termination_status(model) != MOI.OPTIMAL
        #     @error "Trilateration failed."
        #     return nothing
        # end

        # Ignore termination status and continue.

        previousx = currentx
        currentx = value.(x)

        if any(.!isfinite.(currentx))
            @info "Optimization resulted in NaN, terminating."
            currentx = previousx
            break
        end

        if norm(currentx - previousx) / norm(currentx) < tol
            # @info "Change in position is too small (k=$k)."
            break
        end

        tau = min(mu * tau, taumax)
    end

    return currentx
end

# Maximum likelihood estimators.
function trilat_ml(s, d; x0 = vec(mean(s, dims = 2)), maxiter = 1000, tol = 1e-16)
    model(s, x) = vec(sqrt.(sum((x .- s) .^ 2, dims = 1)))
    fit = curve_fit(model, s, d, x0, maxIter = maxiter, x_tol = tol)
    return coef(fit)
end

function trilat_ml_sr_ls(
    s,
    d2,
    w = 1 ./ d2;
    x0 = vec(mean(s, dims = 2)),
    maxiter = 1000,
    tol = 1e-16,
)
    model(s, x) = vec((sum((x .- s) .^ 2, dims = 1)))
    fit = curve_fit(model, s, d2, w, x0; maxIter = maxiter, x_tol = tol)
    return coef(fit)
end

function trilat_ml_rss(
    s,
    C,
    C0,
    eta;
    x0 = vec(mean(s, dims = 2)),
    maxiter = 1000,
    tol = 1e-16,
)
    model(s, x) = 10 .* eta .* log10.(vec(sqrt.(sum((x .- s) .^ 2, dims = 1))))
    fit = curve_fit(model, s, C0 .- C, x0, maxIter = maxiter, x_tol = tol)
    return coef(fit)
end

function trilat_ml_rtt_rss(
    rtt_s,
    d,
    rtt_sigma,
    rss_s,
    C,
    C0,
    eta,
    rss_sigma;
    x0 = vec(mean(s, dims = 2)),
    maxiter = 1000,
    tol = 1e-16,
)
    rtt_m = size(rtt_s, 2)
    rss_m = size(rss_s, 2)
    wt = [fill(1 / rtt_sigma^2, rtt_m); fill(1 / rss_sigma^2, rss_m)]
    model(s, x) = [
        vec(sqrt.(sum((x .- rtt_s) .^ 2, dims = 1)))
        10 .* eta .* log10.(vec(sqrt.(sum((x .- rss_s) .^ 2, dims = 1))))
    ]
    fit = curve_fit(
        model,
        [rtt_s rss_s],
        [d; C0 .- C],
        wt,
        x0,
        maxIter = maxiter,
        x_tol = tol,
    )
    return coef(fit)
end
