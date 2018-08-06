import numpy as np
import pandas as pd
import setting

class Gas_Module():
    def __init__(self, main, P):
        self.P_std = 1e5
        self.main = main
        self.metal_list = main.metal_list
        P_atom = pd.Series(P, setting.Elements_name)
        self.ln_P_element = np.log(P_atom/self.P_std)
        self.P = self.generate_complete_P(P_atom)
        self.full_index = self.P.index
        self.update_from_P()

    def generate_complete_P(self, P):
        total = 0
        p_list = []
        for metal in self.metal_list:
            p = metal.X_P * P[metal.name]
            p = pd.Series(p, setting.isotopes_name[metal.name])
            p[metal.name] = P[metal.name]
            total += P[metal.name]
            p_list.append(p)
        p_list = pd.concat(p_list)
        p_list["n_tot"] = total
        P = p_list[setting.total_isotopes_name + setting.Elements_name + ["n_tot"]]
        return P

    def update_pressure(self, P, ln_P):
        self.P = P[self.full_index]
        self.ln_P_element = ln_P[setting.Elements_name]

    def update_from_P(self):
        self.N_1 = self.P/(setting.k * self.main.T)


    def update_from_N_1(self):
        self.P = self.N_1 * setting.k * self.main.T


    def update_total(self):
        tot = 0
        for name in setting.Elements_name:
            self.P[name] = self.P[setting.isotopes_name[name]].sum()
            tot += self.P[name]
        self.P["n_tot"] = tot
        self.update_from_P()
