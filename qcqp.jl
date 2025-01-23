using LinearAlgebra

# S. Adachi and Y. Nakatsukasa, “Eigenvalue-based algorithm and analysis for nonconvex QCQP with one constraint,” Math. Program., vol. 173, no. 1, pp. 79–116, Jan. 2019, doi: 10.1007/s10107-017-1206-8.
function qcqp(A, a, B, b, beta, lambdahat; inftol = 1e12)
    # Solves min x'*A*x + 2*a'*x
    # subject to x'*B*x + 2*b'*x + beta = 0
    # where A + lambdahat*B is positive definite.

    # See Algorithm 3.2 in the paper.

    n = size(A, 1)

    M0 = [beta b' -a'; b B -A; -a -A zeros(n, n)]
    M1 = [0 zeros(1, n) -b'; zeros(n, 1) zeros(n, n) -B; -b -B zeros(n, n)]
    Mhat = M0 + lambdahat * M1

    xg = -(A + lambdahat * B) \ (a + lambdahat * b)
    gamma = xg' * B * xg + 2 * b' * xg + beta

    x = nothing

    # Tolerance for the constraint x'*B*x + 2*b'*x + beta = 0.
    gammatol = 1e-16
    if abs(gamma) < gammatol
        gamma = 0.0
    end

    if gamma > 0
        (xis, zs) = eigen(M1, -Mhat)
        keep = isreal.(xis) .& isfinite.(xis) .& (real.(xis) .< inftol)
        xis = real(xis[keep])
        zs = real(zs[:, keep])

        (xi, ind) = findmax(xis)
        z = zs[:, ind]

        if xi == 0
            error("ξ = 0, consider increasing lambdahat")
        end

        lambda = lambdahat + 1 / xi
    elseif gamma < 0
        (xis, zs) = eigen(M1, -Mhat)
        keep = isreal.(xis) .& isfinite.(xis) .& (real.(xis) .> -inftol)
        xis = real(xis[keep])
        zs = real(zs[:, keep])

        (xi, ind) = findmin(xis)
        z = zs[:, ind]

        if xi == 0
            error("ξ = 0 , consider increasing lambdahat.")
        end

        lambda = lambdahat + 1 / xi
    else
        # gamma = 0 => the constraint is satisfied and xg is a solution.
        lambda = lambdahat
        x = xg
    end

    if gamma != 0
        theta = z[1]
        y1 = z[2:n+1]

        thetatol = 1e-6
        if abs(theta) > thetatol
            x = y1 / theta
        else
            # TODO: Deal with the "hard case", i.e., when (A + lambda * B) is singular.
            return -(A + lambda * B) \ (a + lambda * b)
        end
    end
    return x
end
