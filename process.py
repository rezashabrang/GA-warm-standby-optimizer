from functions import Process

# ---------------------- Globals ----------------------
T = 210  # Time units of mission
J = 2  # N operation systems
H = 2  # N rescue systems
m = 3  # N intervals
pi = 50  # FDSM paramter
CF = 10000  # Failure Cost
CL = 300000  # System Loss Cost


def cost_func(proposed_system, return_R_SS_C_alt_SS_alt=False, C_MAX=100000000000000, C_MAX_DIVIDER=1):
    global T
    global m
    global pi
    process = Process(proposed_system)
    process.set_globals(T, m, pi)
    # ----------------- Calculating Probs -----------------
    R_val = process.R()
    W_val = process.W()
    SS = R_val + W_val
    CE = process.calculate_equipment_cost()

    C_tot_altered = 0

    # ----------------- Calculating total cost -----------------
    C_tot = CF * (1 - R_val) + CL * (1 - SS) + CE
    if C_MAX <= C_tot:
        C_tot_altered = (C_tot - C_MAX) / C_MAX_DIVIDER
        SS_alt = SS - C_tot_altered
    else:
        SS_alt = SS

    if return_R_SS_C_alt_SS_alt:
        return R_val, SS, C_tot, CE, C_tot_altered, SS_alt

    return SS_alt
