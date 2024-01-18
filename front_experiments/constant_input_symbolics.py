from sympy import simplify, solve, symbols

alpha, beta, c, eps, theta = symbols(r"\alpha \beta c \epsilon \theta")

denom = (
    theta * c / (1 + c) - alpha * beta * c / 2 / (1 + c) / (1 + beta + c * alpha) ** 2
)
asymptotic = c / denom

expr = 2 * theta * (1 + beta + c * alpha) * (1 + c) - (1 + c * alpha)
linearized = expr.diff(theta) / expr.diff(c)

c_subs = solve(expr, c)[1]

assert (linearized.expand() - asymptotic.expand()).subs(
    c, c_subs
).expand().simplify() == 0
