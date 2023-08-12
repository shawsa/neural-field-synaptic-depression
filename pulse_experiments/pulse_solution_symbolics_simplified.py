import sympy as sym

from helper_symbolics import expr_dict, recursive_reduce

OUT_FILE = "simplified_pulse_profile.tex"

r = sym.symbols("r_1")
bsym = sym.symbols([f"B_{i}" for i in range(14)])
alpha, beta, gamma = sym.symbols(r"\alpha, \beta, \gamma")
sub_dict = expr_dict["sub_dict"]

# change to new parameters alpha*beta |--> beta
for key, value in sub_dict.items():
    sub_dict[key] = sub_dict[key].subs(alpha * beta, beta)
expr1, expr2 = expr_dict["speed_width_conditions"]
U = expr_dict["U"]


# easy substitutions simplifications
for key, value in sub_dict.items():
    sub_dict[key] = sub_dict[key].subs(
        bsym[1] + bsym[2], sym.simplify(sub_dict[bsym[1]] + sub_dict[bsym[2]])
    )


for key, value in sub_dict.items():
    sub_dict[key] = sub_dict[key].subs(bsym[2], beta * bsym[1])

del sub_dict[bsym[2]]

# B_1 is just gamma
for key, value in sub_dict.items():
    sub_dict[key] = sub_dict[key].subs(bsym[1], gamma)
U = U.subs(bsym[1], gamma)
expr1 = expr1.subs(bsym[1], gamma)
expr2 = expr2.subs(bsym[1], gamma)
del sub_dict[bsym[1]]


# substitute some intermediates
for sub_index in [5, 7, 8]:
    for key, value in sub_dict.items():
        sub_dict[key] = sub_dict[key].subs(bsym[sub_index], sub_dict[bsym[sub_index]])
        U = U.subs(bsym[sub_index], sub_dict[bsym[sub_index]])
    del sub_dict[bsym[sub_index]]

sub_dict[bsym[11]] = sub_dict[bsym[11]].simplify()

with open(OUT_FILE, "w") as f:
    f.write("\\begin{align*}\n")
    f.write(f"U &= {sym.latex(U)} \\\\ \n")
    for expr in expr1, expr2:
        f.write(f"0 &= {sym.latex(expr)} \\\\ \n")
    for key, value in sub_dict.items():
        f.write(f"{sym.latex(key)} &= {sym.latex(value)} \\\\ \n")
    f.write("\\end{align*}")
