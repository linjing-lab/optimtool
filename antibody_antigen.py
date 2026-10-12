import optimtool.unconstrain as ou
from optimtool.base import sp

# Coordinate variables (antibody and antigen atoms)
x_ab, y_ab, z_ab = sp.symbols('x_ab y_ab z_ab')
x_ag, y_ag, z_ag = sp.symbols('x_ag y_ag z_ag')

# Distance and angle
d = sp.symbols('d')                 # interatomic distance
d0 = sp.symbols('d0')               # target distance (prior)
sigma = sp.symbols('sigma')         # distance tolerance

# Contact energy parameters
epsilon = sp.symbols('epsilon')     # contact well depth
r_cut = sp.symbols('r_cut')         # contact cutoff radius

# Clash penalty parameters
d_min = sp.symbols('d_min')         # minimum allowed distance
k_clash = sp.symbols('k_clash')     # clash penalty coefficient

# Structural prior
theta = sp.symbols('theta')         # dihedral / orientation angle
theta0 = sp.symbols('theta0')       # ideal orientation angle
k_angle = sp.symbols('k_angle')     # angle restraint coefficient

# 1. Distance restraint term (harmonic restraint)
distance_term = (d - d0)**2 / (2 * sigma**2)

# 2. Interface contact potential (Lennard-Jones-like soft well)
contact_term = epsilon * ((r_cut**12 / (d**12 + 1e-6)) - (r_cut**6 / (d**6 + 1e-6)))

# 3. Steric clash penalty (soft-core repulsion)
clash_term = k_clash * sp.exp(-(d - d_min))

# 4. Structural orientation prior (angle restraint)
angle_term = k_angle * (1 - sp.cos(theta - theta0))

f = distance_term + contact_term + clash_term + angle_term

struct_vars = [x_ab, y_ab, z_ab, x_ag, y_ag, z_ag]  # antibody atom xyz + antigen atom xyz
params = [d, d0, sigma, epsilon, r_cut, d_min, k_clash, theta, theta0, k_angle]
all_symbols = struct_vars + params

# [antibody xyz (Å), antigen xyz (Å), d, d0, sigma, epsilon, r_cut, d_min, k_clash, theta, theta0, k_angle]
initial_point = [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 6.0, 5.5, 0.5, 1.2, 8.0, 3.5, 10.0, 1.0, 0.9, 2.0]

ou.gradient_descent.barzilar_borwein(f, all_symbols, initial_point, verbose=True, epsilon=1e-4)