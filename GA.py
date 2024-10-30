from GA_helper import generate_system, crossover, decode_system, encode_sytem, parent_selection, mutation, \
    strategy_based_cost_update
from process import cost_func
import random
from tqdm import tqdm
from pprint import pprint

# ---------------------- GLOBALS ----------------------
n_pop = 5
max_iteration = 1
pc = 0.8  # Crossover percentage
nc = 2 * round(pc * n_pop / 2)

pm = 0.4  # Mutation percentage
nm = round(pm * n_pop)  # Number of mutants

mu = 0.2  # Mutation Rate

# CROSSOVER METHODS: single-point, double-point, uniform
crossover_method = "uniform"

# PARENT SELECTION METHODS: roulette , tournament, random
ps_method = "tournament"
k = 2  # if tournament method is selected

# Probability penalty terms
C_MAX = 10000
C_MAX_DIVIDER = 1000

pop = [generate_system() for system in tqdm(range(n_pop), desc="Population Generation")]  # Generating population

# Best cost based on strategy
for idx, genome in enumerate(pop):
    new_cost = strategy_based_cost_update(genome["system"], C_MAX=C_MAX, C_MAX_DIVIDER=C_MAX_DIVIDER)
    pop[idx]["cost"] = new_cost

for iteration in tqdm(range(max_iteration), desc="Main Iteration"):
    children = []
    # --------------------- CROSSOVER ---------------------
    crossover_parents = parent_selection(pop, nc, ps_method, k)
    encoded_parents = [encode_sytem(parent) for parent in crossover_parents]
    for i in range(0, nc, 2):
        p1, p2 = encoded_parents[i], encoded_parents[i + 1]
        c1, c2 = crossover(p1, p2, crossover_method)
        c1, c2 = decode_system(c1), decode_system(c2)
        c1["cost"], c2["cost"] = cost_func(c1["system"], C_MAX=C_MAX, C_MAX_DIVIDER=C_MAX_DIVIDER), cost_func(
            c2["system"], C_MAX=C_MAX, C_MAX_DIVIDER=C_MAX_DIVIDER)
        children.append(c1)
        children.append(c2)

    # --------------------- MUTATION ---------------------
    mutation_parents = parent_selection(pop, nm, ps_method, k)
    encoded_parents = [encode_sytem(parent) for parent in mutation_parents]
    for j in range(nm):
        par = encoded_parents[j]
        child = mutation(par, mu)
        child = decode_system(child)
        child["cost"] = cost_func(child["system"], C_MAX=C_MAX, C_MAX_DIVIDER=C_MAX_DIVIDER)
        children.append(child)

    # Adding children to population and cutting the worst results.
    for ch in children:
        pop.append(ch)

    # ------------------- Updating cost based on strategy -------------------
    for idx, genome in enumerate(pop):
        new_cost = strategy_based_cost_update(genome["system"], C_MAX=C_MAX, C_MAX_DIVIDER=C_MAX_DIVIDER)
        pop[idx]["cost"] = new_cost

    pop = sorted(pop, key=lambda system: system["cost"], reverse=True)
    pop = pop[:n_pop]

best_cost, best_sys, R, SS, CE, C_tot_alt, SS_alt = strategy_based_cost_update(pop[0]["system"], return_R_SS_CE=True,
                                                                               C_MAX=C_MAX, C_MAX_DIVIDER=C_MAX_DIVIDER)

print("BEST SYSTEM:")
pprint(best_sys)
print("BEST COST:")
print(best_cost)
print("BEST R:")
print(R)
print("BEST SS:")
print(SS)
print("BEST CE:")
print(CE)
print("BEST Altered C Total:")
print(C_tot_alt)
print("BEST Altered SS:")
print(SS_alt)
