import setting
import pandas as pd
import unicon
from unicon_utils import *
from pre_process import *
from g_mix import *

class Gas_Equilibrium_Module():
    ## The unit here is atm
    def __init__(self, main, p_gas_tot, elements, P_tot = None):
        self.Gas_Elements_name = elements
        self.main = main
        if P_tot is not None:
            self.P_tot = P_tot
        else:
            self.P_tot = pd.Series(p_gas_tot, elements)
            self.P_tot = pd.concat((self.P_tot, main.Gas.P))/1e5
        unicon.initialization()
        unicon.T = 1700
        for element in self.Gas_Elements_name + setting.Elements_name:
            unicon.Element(element, self.P_tot[element], self.P_tot[element]/self.P_tot["H"]*2.43e10)

        add_species("H", s_type = "monoatom")
        add_species("H2", {"H":2}, s_type = "gas")
        add_species("H2O", {"H":2, "O":1}, s_type = "gas")

        add_species("O", s_type = "monoatom")
        add_species("O2", {"O":2},s_type = "gas")

        add_species("C", s_type = "monoatom")
        add_species("CO1", {"C":1, "O":1},s_type = "gas")

        add_species("Fe", s_type = "monoatom")
        add_species("Ni", s_type = "monoatom")
        add_species("Ir", s_type = "monoatom")
        add_species("Mo", s_type = "monoatom")
        add_species("MoO", {"Mo":1,"O":1},s_type = "gas")
        add_species("MoO2", {"Mo":1, "O":2},s_type = "gas")
        add_species("MoO3", {"Mo":1, "O":3},s_type = "gas")

        add_species("Ru", s_type = "monoatom")
        add_species("Pt", s_type = "monoatom")
        add_species("Os", s_type = "monoatom")
        add_species("Re", s_type = "monoatom")

        add_species("W", s_type = "monoatom")
        add_species("WO", {"W":1, "O":1},s_type = "gas")
        add_species("WO2", {"W":1, "O":2},s_type = "gas")
        add_species("WO3", {"W":1, "O":3},s_type = "gas")
        add_species("W3O8", {"W":3, "O":8},s_type = "gas")
        add_species("W3O9", {"W":3, "O":9},s_type = "gas")

        x_0 = np.array([7.5e-3*self.P_tot["H"], 6e-6*self.P_tot["O"], 4e-1*self.P_tot["C"], \
                        self.P_tot["Fe"], self.P_tot["Ni"], self.P_tot["Ir"], 2e-1*self.P_tot["Mo"], \
                        self.P_tot["Ru"], self.P_tot["Pt"], self.P_tot["Os"], 2e-4*self.P_tot["W"],\
                       self.P_tot["Re"]])
        x_0 = np.log(x_0)
        self.species_name = self.Gas_Elements_name + setting.Elements_name
        self.Monoatom_name = list(unicon.Monoatom_dict.keys())
        encode_x(x_0, self.species_name)
        self.x = extract_x()
        self.x = update_state(self.x)
        for t in np.arange(unicon.T, main.T + np.sign(main.T - unicon.T), np.sign(main.T - unicon.T)):
            unicon.T = t
            self.x = update_state(self.x)

        # save the address of WO gas
        for gas in unicon.Gas_species_list:
            if gas.name == "WO":
                self.WO_gas = gas

    def delta_gas(self, delta):
        # delta has unit of m^(-3)
        idx = delta.index
        p_delta = 1e-5 * delta * setting.k * self.main.T
        # mask = p_delta < self.P_tot[idx]
        idx = idx[p_delta < self.P_tot[idx]]
        self.P_tot.loc[idx] -= p_delta[idx]
#         self.P_atom[idx] -= delta
#         self.encode_x()
        for element in setting.Elements_name:
            unicon.Elements[element].p_tot = max(self.P_tot[element], 1e-50)

    def get_pressure(self):
        self.ln_P_atom = pd.Series(self.x, self.Monoatom_name)[setting.Elements_name]
        self.P_collied =  np.exp(self.ln_P_atom)
        self.P_collied["W"] += self.WO_gas.pressure
        self.P_collied["n_tot"] = np.sum(self.P_collied)
        for element in setting.Elements_name:
            self.P_collied = self.P_collied.append(self.P_tot[setting.isotopes_name[element]] *self.P_collied[element]/self.P_tot[element])
        self.P_collied = self.P_collied[setting.total_isotopes_name + setting.Elements_name + ["n_tot"]]
        return 1e5*self.P_collied, self.ln_P_atom

    def encode_x(self):
        X = self.ln_P_atom.apply(lambda x: max(x, -115))
        encode_x(X.values, X.index)

    def update_T(self):
        unicon.T = self.main.T
        self.x = update_state(self.x)
