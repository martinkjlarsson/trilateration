using LinearAlgebra

"""
    trilat(s, d2, W = Diagonal(1 ./ d2); maxiter = 0)

Solves the trilateration problem using Algorithm 2 in the paper. Formally, we
are minimizing `∑ⱼwⱼ(||x - sⱼ||² - dⱼ²)²` over `x`, where `sⱼ` are
the columns of `s`, `dⱼ²` are the elements of `d2`, and `wⱼ` are the diagonal
elements of `W`. Local optimization is performed if `maxiter > 0`.

Correlated distance measurements `dⱼ` are modelled by providing a nondiagonal
`W`. The objective function is then given by
`∑ᵢⱼwᵢⱼ(||x - sᵢ||² - dᵢ²)(||x - sⱼ||² - dⱼ²)`.

A matrix with one or two columns are return corresponding to the one or two
solutions `x` to the problem.
"""
function trilat(
    s,
    d2,
    W::Union{Diagonal,Symmetric,SymTridiagonal} = Diagonal(1 ./ d2);
    maxiter = 0,
)
    n = size(s, 1)

    # Normalize weights.
    W = W ./ sum(W)

    # Translate coordinate system.
    t = s * sum(W, dims = 2)
    st = s .- t

    # Construct A and g such that (x'*x)*x - A*x + g = 0.
    ws2md2 = W * (vec(sum(abs2, st, dims = 1)) - d2)
    A = Symmetric(-2 * st * W * st' - sum(ws2md2) * I)
    g = -st * ws2md2

    # Rotate coordinate system.
    (D, Q) = eigen(A, sortby = λ -> -real(λ))
    b = Q' * g

    # We now have D and b such that (y'*y)*y - diagm(D)*y + b = 0.

    # Constuct M and find largest real eigenvalue.
    M = [
        diagm(D) diagm(-b) zeros(n, 1)
        zeros(n, n) diagm(D) -b
        ones(1, n) zeros(1, n + 1)
    ]
    lambdas = eigvals(M, sortby = λ -> -real(λ))
    λmax = real(lambdas[1])

    rnk = rank(Diagonal(λmax .- D), rtol = n * sqrt(eps()))

    # Find receiver position.
    if rnk == n
        y = [0; -b[2:end] ./ (λmax .- D[2:end])]
        y[1] = -sign(b[1]) * sqrt(max(λmax - dot(y, y), 0))
    elseif rnk == n - 1
        y = [0; -b[2:end] ./ (λmax .- D[2:end])]
        y1 = sqrt(max(λmax - dot(y, y), 0))
        y = [y y]
        y[1, 1] = y1
        y[1, 2] = -y1
    else
        @warn "Trilateration did not have finitely many solutions."
        return nothing
    end

    # Local optimization.
    if maxiter > 0
        for i in axes(y, 2)
            yi = y[:, i]
            for _ in 1:maxiter
                res = (yi' * yi) * yi - Diagonal(D) * yi + b
                J = (yi' * yi) * I + 2 * (yi * yi') - Diagonal(D)
                yi = yi - J \ res
            end
            y[:, i] = yi
        end
    end

    # Revert transformations.
    x = Q * y .+ t

    return x
end

"""
    trilat_A(s, d2, W = Diagonal(1 ./ d2))

Solves the trilateration problem using Algorithm 1 in the paper.
"""
function trilat_A(s, d2, W = Diagonal(1 ./ d2))
    n = size(s, 1)

    # Normalize weights.
    W = W ./ sum(W)

    # Translate coordinate system.
    t = s * sum(W, dims = 2)
    st = s .- t

    # Construct A and g such that (x'*x)*x - A*x + g = 0.
    ws2md2 = W * (vec(sum(abs2, st, dims = 1)) - d2)
    A = -2 * st * W * st' - sum(ws2md2) * I
    g = -st * ws2md2

    # Constuct M and find largest real eigenvalue.
    Ma = [
        A I zeros(n, 1)
        zeros(n, n) A -g
        -g' zeros(1, n + 1)
    ]
    lambdas = eigvals(Ma, sortby = λ -> -real(λ))
    λmax = real(lambdas[1])

    # Find receiver position. Note that rank deficient cases are not treated.
    x = -(λmax * I - A) \ g

    # Revert translation.
    x += t

    return x
end
