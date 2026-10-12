from optimtool.base import sp
import optimtool.unconstrain as ou

# RFdiffusion / AF2 / Boltz
pLDDT, ipTM, iPAE, RMSD = sp.symbols("pLDDT ipTM iPAE RMSD", real=True)
# ProteinMPNN
mpnn_nll = sp.symbols("mpnn_nll", real=True)
# Rosetta / PRODIGY
ddG, hyd_patch = sp.symbols("ddG hyd_patch", real=True)

x = sp.Matrix([pLDDT, ipTM, iPAE, RMSD, mpnn_nll, ddG, hyd_patch])
beta = 4.0
softplus = lambda z: sp.log(1 + sp.exp(beta * z)) / beta

f = (
    1.00 * softplus((80   - pLDDT)/10)**2 +
    1.20 * softplus((0.65 - ipTM)/0.2)**2 +
    0.80 * softplus((iPAE - 10)/5)**2 +
    0.60 * softplus((RMSD - 2.0)/1.0)**2 +
    0.90 * softplus((mpnn_nll - 1.15)/0.15)**2 +
    1.00 * softplus((ddG + 12)/3)**2 +
    0.50 * softplus((hyd_patch - 0.25)/0.15)**2
)
x0 = (72.0, 0.55, 14.0, 3.5, 1.35, -6.0, 0.35)
ou.gradient_descent.barzilar_borwein(f, x, x0, verbose=True, epsilon=1e-4)
# (array([ 8.87545103e+01,  1.02715306e+00,  4.87408782e+00,  6.04287489e-01, 8.61737702e-01, -1.55498465e+01, -2.72336446e-02]), 19)