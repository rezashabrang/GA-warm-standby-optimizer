import math
from math import pow, exp
import sys

from SYSTEM import SPECS
from functools import reduce
from itertools import product
from math import prod

from scipy.integrate import quad


# -------------- Constants (globals) --------------
class Process:
    def __init__(self, SYSTEM):

        # Duration of simulation
        self.T = 0

        # Number of intervals
        self.m = 1

        # Delta
        self.delta = 1

        # FDSM Param
        self.pi_fdsm = 99500

        self.PROPOSED_SYSTEM = SYSTEM

    def set_globals(self, t_val, m_val, pi):
        self.T = t_val
        self.m = m_val
        self.delta = self.T / self.m
        self.pi_fdsm = pi

    def calculate_equipment_cost(self):
        """
        Calculating equipment Cost
        """
        total_cost = 0
        for key, val in self.PROPOSED_SYSTEM.items():
            spec = SPECS[key]
            for i in val:
                if type(i) == list:
                    for j in i:
                        total_cost += spec[j]['c']
                else:
                    total_cost += spec[i]['c']
        return total_cost

    def torque(self,t):
        """
        duration of successful rescue procedure when it is activated at time t
        args:
            t: Entire time from the entire mission begining
        """
        if t % 70 == 0:
            return 5
        t_c = t - 70 * math.floor(t / 70)
        res = 5 + 45 / (1 + math.pow(0.05 * t_c, -4.2))
        return res

    def mu(self, v):
        """
        """
        return self.torque(v * self.delta) / self.delta

    def FDSM(self, t):
        """
        failure detection and switching mechanism probability
        args:
            t: Entire time from the entire mission begining
        """
        res = 1 - math.exp(-t / self.pi_fdsm)
        return res

    def Z(self, i):
        """
    
        """
        res_Z = self.FDSM(self.delta * i)
        if res_Z < 0:
            return 0
        return res_Z

    def F_Weibull(self, t, j, k):
        """
        Weibull time-to failure cumulativce
        args:
            t: Entire time from the entire mission beginning
            j: Subsystem type
            k: Component type
        """
        # Defining Constants
        n_j = SPECS[j][k]['n']
        b_j = SPECS[j][k]['b']
        res = 1 - math.exp((-1) * math.pow(t / n_j, b_j))
        return res

    def f_weibull(self, t, j, k):
        """
        Weibull time-to failure density
        args:
            t: Entire time from the entire mission begining
            j: Subsystem type
            k: Component type
        """
        # Defining Constants
        n_j = SPECS[j][k]['n']
        b_j = SPECS[j][k]['b']
        res = (b_j / n_j) * pow((t / n_j), b_j - 1) * exp(-pow((t / n_j), b_j))
        return res

    def f_max(self, t, j, k, group_size):
        """Order statistics density
        args:
            t: time,
            j: system type,
            k: type of component
            group_size: count of components grouped together
        """
        res = group_size * self.f_weibull(t, j, k) * pow(self.F_Weibull(t, j, k), group_size - 1)
        return res

    def F_MAX(self, t, j, k, group_size):
        """Order statistics cumulative
        """
        res = quad(self.f_max, 0, t, args=(j, k, group_size))
        return res

    def p_hat(self, i_s, i_o, j, k):
        """
        probability that component of type component_type that should be activated
        in interval i_s fails before functioning for i_o time intervals
        args:
            i_s: Standby time interval,
            i_o: Opertaion time interval,
            j: System Type,
            k: Type of component
        """
        if type(k) == list:
            d_j = SPECS[j][k[0]]['d']
            type_k = k[0]  # Since it is homogenous
            group_size = len(k)  # How many components are grouped
            t = self.delta * (d_j * i_s + i_o)
            integram_vals = self.F_MAX(t, j, type_k, group_size)
            p_hat_val = integram_vals[0] - integram_vals[1]
            return p_hat_val

        else:
            # Defining Constant
            d_j = SPECS[j][k]['d']
            return self.F_Weibull(self.delta * (d_j * i_s + i_o), j, k)

    def p(self, i_s, i_o, j, k):
        """
        probability that component of type k that should be activated
        in interval i_s fails after functioning for exactly i_o time interval
        args:
            i_s: Standby time interval,
            i_o: Opertaion time interval,
            j: System Type,
            k: Type of component
        """
        if type(k) == list:
            # ------------------- Creating p_hat of all subcomponents matrix -------------------
            # Initializing p_hat matrix
            p_mat = []
            # For every component in the group of the component
            for sub_component in k:
                # Temporary matrix for every sub component
                temp_p_mat = []
                # For every operation time less than or equal of current operation time.
                for operation_time in range(i_o + 1):
                    temp_p_mat.append(
                        self.p_hat(i_s, operation_time + 1, j, sub_component) - self.p_hat(i_s, operation_time, j, sub_component))
                p_mat.append(temp_p_mat)

            # ------------------- Calculating final p_hat value for group -------------------~
            final_p = 0
            all_working_prob = 0  # Probability when all devices are working in the time interval.
            for c_index, component in enumerate(p_mat):
                # List used for product
                temp = [[component[-1]]]

                # Creating matrix for product calculation
                for i in range(len(p_mat)):
                    if i == c_index:
                        continue
                    temp.append(p_mat[i])

                combinations = list(product(*temp))  # Computing combinations
                # Calculating probabilities
                all_working_prob = reduce(lambda x, y: x * y, combinations[-1])
                combinations.pop()
                for comb in list(combinations):
                    final_p += prod(comb)

            final_p += all_working_prob  # Adding the final prob

            return final_p
        else:
            return self.p_hat(i_s, i_o + 1, j, k) - self.p_hat(i_s, i_o, j, k)

    def Q(self, j, k_j, x, k):
        """
        pmf of the random variable X(j,k) which represents the time interval that component k
        from subsystem j fails.
        args:
            j: Subsystem index
            k_j: Component index
            x: Time interval index
            k: Type of component
        """
        # First defining the initial values
        if k_j == 0:
            if x == 0:
                return 1
            else:
                return 0

        # Defining previous component type
        k = self.PROPOSED_SYSTEM[j][k_j - 1]

        # Calculating terms
        first_term = self.Q(j, k_j - 1, x, k) * self.p_hat(x, 0, j, k)
        second_term = 0
        for y in range(x):
            second_term += self.Q(j, k_j - 1, y, k) * self.p(y, x - y, j, k)

        return first_term + second_term

    def U(self, j, i):
        """
        The probability that subsystem j fails in no later than time
        interval i.
        args:
            j: Subsystem index
            i: Time interval
            K_j: Last component of subsystem j
            x: Time interval
            k: Component Type
        """
        # Getting the last component specs of the subsystem j
        K_j = len(self.PROPOSED_SYSTEM[j])  # Getting the index (MAX should be four)
        k = self.PROPOSED_SYSTEM[j][-1]
        res = 0
        for x in range(i):
            res += self.Q(j, K_j, x, k)
        return res

    def U_OS(self, i):
        """
        probability that the entire operation system fails in no later than time interval i
        args:
            i: Time interval index
            J: Operation subsystems
        """
        J = 2
        res = 1
        for j in range(1, J + 1):
            res *= (1 - self.U(j, i))
        return 1 - res

    def Q_OS(self, i):
        """
        The pmf of the discrete time-to-failure of the entire operation system.
        args:
            i: Time interval index
        """
        return self.U_OS(i) - self.U_OS(i - 1)

    # -------------------------------------------- MSP --------------------------------------------
    def PR_e_j_k(self, j, k_j):
        """
        Probability of event e_j,k that component s_j(k) successfully accomplishes the mission task.
        """
        k = self.PROPOSED_SYSTEM[j][k_j - 1]
        res = 0
        for x in range(self.m):
            res += self.Q(j, k_j - 1, x, k) * (1 - self.p_hat(x, self.m - x, j, k)) * 0.99

        return res

    def r(self, j):
        """
        The probability that subsystem j can successfully accomplish the mission.
        Args:
            j: Subsystem index
        """
        # Defining K_j
        K_j = len(self.PROPOSED_SYSTEM[j])

        res = 0
        for k_j in range(1, K_j + 1):
            res += self.PR_e_j_k(j, k_j)
        return res

    def R(self):
        """
        MSP
        """
        J = 2
        res = 1
        for j in range(1, J + 1):
            res *= self.r(j)

        return res

    # -------------------------------------------- SS --------------------------------------------

    def Q_SS(self,j, k_j, v, x, k=0):
        """
        Q for SS
        args:
            j: Subsystem index
            k_j: Component index
            x: Time interval index
            k: Type of component
        """
        # First defining the inital values
        if k_j == 0:
            if x == v:
                return 1
            else:
                return 0
        else:
            if x < v:
                return 0
            elif v <= x:
                # Defining previous component type
                if k_j - 1 <= 0:
                    curr_k = self.PROPOSED_SYSTEM[j][k_j - 1]
                    prev_k = curr_k
                else:
                    prev_k = self.PROPOSED_SYSTEM[j][k_j - 2]
                    curr_k = self.PROPOSED_SYSTEM[j][k_j - 1]

                # Calculating terms
                # changed here
                first_term = self.Q_SS(j, k_j - 1, v, x, prev_k) * self.p_hat(v, 0, j, curr_k)
                second_term = 0
                for y in range(v, x + 1):
                    second_term += self.Q_SS(j, k_j - 1, v, y, prev_k) * self.p(y, x - y, j, curr_k)

                return first_term + second_term
            else:
                return 1

    def PR_e_j_k_SS(self, j, k_j, v):
        """
        SS PR_e_j_k
        Args:
            k_j: Index of component,
            v: Time interval
        """
        k = self.PROPOSED_SYSTEM[j][k_j - 1]
        res = 0
        for x in range(v, self.m):
            # changed here (keep in mind)
            i_o = v + self.mu(v) - x
            if i_o < 0:
                i_o = 0
            res += self.Q_SS(j, k_j - 1, v, x) * (1 - self.p_hat(x, i_o, j, k))

        return res

    def w(self, j, v):
        """
        Probability that rescue system activated in time interval v
        completes the rescue procedure.
        Args:
            j: Subsystem index
            v: Time interval
        """
        subsystem = self.PROPOSED_SYSTEM[j]
        K_j = subsystem.index(subsystem[-1]) + 1
        res = 0
        for k_j in range(1, K_j + 1):
            res += self.PR_e_j_k_SS(j, k_j, v)
        return res

    def W(self):
        """
        Overall success probability of the rescue procedure
        """
        res = 0
        # if no rescue system is provided then return 0 for this part
        if self.PROPOSED_SYSTEM[3] == [] and self.PROPOSED_SYSTEM[4] == []:
            return 0
        for v in range(self.m):
            w_res = 1
            for j in [3, 4]:
                # If there is no rescue system then skip it
                if self.PROPOSED_SYSTEM[j] == []:
                    continue
                w_res *= self.w(j, v)
            res += self.Q_OS(v) * (1 - self.Z(v)) * w_res

        return res

    def SS(self):
        """
        System survivability
        """
        return self.R() + self.W()


    def calculate_equipment_cost(self):
        """
        Calculating equipment Cost
        """
        total_cost = 0
        for key, val in self.PROPOSED_SYSTEM.items():
            spec = SPECS[key]
            for i in val:
                if type(i) == list:
                    for j in i:
                        total_cost += spec[j]['c']
                else:
                    total_cost += spec[i]['c']
        return total_cost