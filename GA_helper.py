from random import randint, random, sample
from process import cost_func
import numpy as np
from process import cost_func
from pprint import pprint


def generate_system():
    sample_system = {}
    for i in range(1, 5):
        # Generating number of components
        n_comp = randint(1, 4) if i in [1, 2] else randint(0, 4)
        # Generating type of components
        component_type = randint(1, 4)
        sample_system[i] = [component_type for _ in range(n_comp)]

    # If subsystem 3 is empty but 4 is not swap them
    if not sample_system[3] and sample_system[4]:
        sample_system[3] = sample_system[4]
        sample_system[4] = []

    # Calculating Cost
    cost = cost_func(sample_system)

    system_info = {
        "system": sample_system,
        "cost": cost
    }
    return system_info


def parent_selection(pop, n, method, k=3):
    """Selecting parents"""
    parents = []
    if method == "random":
        parents = sample(pop, n)
    elif method == "tournament":
        for _ in range(n):
            group = sample(pop, k)
            group = sorted(group, key=lambda system: system["cost"])
            parents.append(group[0])
    elif method == "roulette":
        pop_fitness = sum([1 / system["cost"] for system in pop])
        probabilites = [(1 / system["cost"]) / pop_fitness for system in pop]
        parents = np.random.choice(pop, n, p=probabilites)

    return parents


def crossover(
        p1: list,
        p2: list,
        method: str
):
    while True:
        c1, c2 = p1.copy(), p2.copy()
        if method == "single-point":
            # select crossover point that is not on the end of the string
            pt = randint(1, len(c1) - 2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        elif method == "double-point":
            pt1 = randint(0, 2)
            pt2 = randint(pt1 + 1, 3)
            # perform crossover
            c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
            c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
        elif method == "uniform":
            for i in range(4):
                if random() <= 0.5:
                    temp = c1[i]
                    c1[i] = c2[i]
                    c2[i] = temp

        # Check children if Main subsystem are empty continue
        if not c1[0] or not c1[1] or not c2[0] or not c2[1]:
            continue
        else:
            break
    return c1, c2


def mutation(par, rate):
    for i in range(4):
        if random() < rate:
            comp_count = randint(1, 4) if i in [0, 1] else randint(0, 4)
            comp_type = randint(1, 4) if comp_count != 0 else 0
            par[i] = [comp_type, comp_count]

    if par[2] == [0, 0] and par[3] != [0, 0]:
        par[2] = par[3]
        par[3] = [0, 0]

    return par


def encode_sytem(system):
    encoded_sys = [0] * 4
    for i in range(1, 5):
        component_type = 0 if not system["system"][i] else system["system"][i][0]
        encoded_sys[i - 1] = [component_type, len(system["system"][i])]

    return encoded_sys


def decode_system(system):
    decoded_sys = {}
    for id, item in enumerate(system):
        decoded_sys[id + 1] = [item[0] for _ in range(item[1])]

    return {"system": decoded_sys}


def strategy_based_cost_update(system, return_R_SS_CE=False, C_MAX=100000000000000000, C_MAX_DIVIDER=1):
    """Checking grouping for subsystem"""
    best_cost = 1000000000000
    best_R = 0
    best_SS = 0
    best_CE = 0
    best_C_tot_alt = 0
    best_SS_alt = -100000000000000000
    best_sys = {}
    subsystem_combination = {}

    # --------- Possible Subsystem grouping combination ---------
    for sub_idx in range(1, 5):
        subsystem = system[sub_idx]
        n_comp = system[sub_idx]
        if len(n_comp) in [0, 1]:
            subsystem_combination[sub_idx] = [subsystem]
        elif len(n_comp) == 2:
            subsystem_combination[sub_idx] = [subsystem]
            subsystem_combination[sub_idx].append([subsystem])
        elif len(n_comp) == 3:
            subsystem_combination[sub_idx] = [subsystem]
            subsystem_combination[sub_idx].append([[subsystem[0], subsystem[1]], subsystem[2]])
            subsystem_combination[sub_idx].append([subsystem])
        elif len(n_comp) == 4:
            subsystem_combination[sub_idx] = [subsystem]
            subsystem_combination[sub_idx].append([[subsystem[0], subsystem[1]], subsystem[2], subsystem[3]])
            subsystem_combination[sub_idx].append([[subsystem[0], subsystem[1], subsystem[2]], subsystem[3]])
            subsystem_combination[sub_idx].append([[subsystem[0], subsystem[1]], [subsystem[2], subsystem[3]]])
            subsystem_combination[sub_idx].append([subsystem])

    for first_sub in subsystem_combination[1]:
        for second_sub in subsystem_combination[2]:
            for third_sub in subsystem_combination[3]:
                for fourth_sub in subsystem_combination[4]:
                    system = {
                        1: first_sub,
                        2: second_sub,
                        3: third_sub,
                        4: fourth_sub
                    }
                    R, SS, cost, CE, C_tot_altered, SS_alt = cost_func(system, return_R_SS_C_alt_SS_alt=True,
                                                                       C_MAX=C_MAX, C_MAX_DIVIDER=C_MAX_DIVIDER)
                    if SS_alt > best_SS_alt:
                        best_R = R
                        best_SS = SS
                        best_cost = cost
                        best_sys = system
                        best_CE = CE
                        best_C_tot_alt = C_tot_altered
                        best_SS_alt = SS_alt

    if return_R_SS_CE:
        return best_cost, best_sys, best_R, best_SS, best_CE, best_C_tot_alt, best_SS_alt

    return best_SS_alt

# PROPOSED_SYSTEM = {
#     1: [4, 4, 4, 4],
#     2: [4, 4, 4],
#     3: [4,4],
#     4: []
# }
#
# print(strategy_based_cost_update(
#     PROPOSED_SYSTEM
# ))
