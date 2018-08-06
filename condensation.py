import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import setting
from scipy.optimize import fsolve


class Nucleus_Growth_Module():
    def __init__(self, main, Ks = np.array([0.61, 1e-7]), Jdt_threshold = 1e-7):
        self.df_nucleus = pd.DataFrame({}, columns = setting.total_isotopes_name \
            + setting.Elements_name + ["n_tot", "N", "volume", "radius", "surface"])
        #Ks is the guess parameters for steady state
        self.main = main
        self.metal_list = main.metal_list
        self.Gas = main.Gas
        self.Ks = Ks
        self.Jdt = 0
        self.Jdt_threshold = Jdt_threshold
        self.update_flag = True

        #generate critical radius
        # Jdt/rou = dr
        # maxRadius = 0
        # for metal in self.metal_list:
        #     maxRadius = max(maxRadius, metal.radius)
        # self.critical_N_density = 1.0/(4/3*math.pi*(maxRadius)**3)


        # self.predict_steady_state_X_isotopes(T)
#         self.K_lower_bound = []
#         for element in Elements_name:
#             self.K_lower_bound.append(1e-20 * self.cond_steady[isotopes_name[element]].sum())
#         self.K_lower_bound = pd.Series(self.K_lower_bound, Elements_name)

    # def update_P(self, P, T):
    #     self.N_1_isotopes = P[setting.total_isotopes_name]/(setting.k*setting.T)
    #
    # def update_N_1(self, N_1):
    #     self.N_1_isotopes = N_1[setting.total_isotopes_name]

    def generate_nucleus(self):
#         X, self.Ks = predict_steady_state_X_isotopes(self.Ks)
        self.df_nucleus.loc[len(self.df_nucleus)] = 100 * self.X_steady
        self.df_nucleus.loc[len(self.df_nucleus) -1, ["N"]] = self.Jdt
        self.Jdt = 0
        self.update_flag = True

    def grow(self, dt):
        grad = self.net_growth()
        growth = self.Integrate_through_t(grad, dt)
        self.df_nucleus[growth.columns] += growth
        self.update_flag = True
        return growth

    def predict_steady_state_X_isotopes(self, T):
        isotopic_mass = np.concatenate([metal.mass_isotopes for metal in self.metal_list])
        isotopic_mass = pd.Series(isotopic_mass, setting.total_isotopes_name)
        sqrt_mass = np.sqrt(2*math.pi*isotopic_mass*setting.k * T)
        cond = self.Gas.P[setting.total_isotopes_name].div(sqrt_mass, axis = 0)
        evap = pd.Series(np.zeros(len(cond)), setting.total_isotopes_name)
        for metal in self.metal_list:
            evap[setting.isotopes_name[metal.name]] = metal.p_sat
        evap = evap[setting.total_isotopes_name].div(sqrt_mass, axis = 0)
        self.Ks = fsolve(func_K_steady_state, self.Ks, args = (1e-8*cond, 1e-8*evap), fprime = func_K_steady_state_prime)
        self.X_steady = x_gamma_K(cond, evap, self.Ks[0], 1e8*self.Ks[1])
        tot = 0
        for name in setting.Elements_name:
            self.X_steady[name] = np.sum(self.X_steady[setting.isotopes_name[name]])
            tot += self.X_steady[name]
        self.X_steady["n_tot"] = tot

    def net_growth(self):
        grad = 1e8*self.Ks[1] * self.X_steady
#         for element in Elements_name:
#             if self.K_lower_bound[element] > grad[element]:
#                 grad[isotopes_name[element] + [element]] = 0
#         grad["n_tot"] = grad[Elements_name].sum()
        grad = grad.values[np.newaxis,:]* self.df_nucleus["surface"].values[:,np.newaxis]
        grad = pd.DataFrame(grad, columns = self.X_steady.index)
        self.grad = grad
        return grad

    def condensation_and_evaporation_for_steady_state(self, T):
        """ This function would return the condenstion and evaporation rate for each isotopes"""
        n_tot = self.df_nucleus["n_tot"]
        X_grain = np.repeat(self.X_steady.values[np.newaxis,:], len(self.df_nucleus), axis = 0)
        X_grain = pd.DataFrame(X_grain, columns = setting.total_isotopes_name)
        ## To save the computation drop the ((n_tot-1)/n_tot)**(2/3) for evaporation
        surface = self.df_nucleus["surface"]
        evap_list = []
        cond_list = []
        for metal in self.metal_list:
    #         p_cond = P[metal.name]
    #         P_evap = metal.p_sat
            name_list = setting.isotopes_name[metal.name]
            P_evap = X_grain[name_list].mul(metal.p_sat)
            sqrt_mass = np.sqrt(2*math.pi*metal.mass_isotopes*setting.k * T)
            evap_rate = P_evap.div(sqrt_mass, axis = 1).mul(surface, axis = 0)
            evap_rate[metal.name] = evap_rate.sum(axis = 1)
            cond_rate = self.Gas.P[name_list]/sqrt_mass[np.newaxis,:] * surface[:, np.newaxis]
            cond_rate = pd.DataFrame(cond_rate, columns = name_list)
            cond_rate[metal.name] = cond_rate.sum(axis = 1)
            evap_list.append(evap_rate)
            cond_list.append(cond_rate)
        cond = pd.concat(cond_list, axis = 1)[setting.total_isotopes_name + setting.Elements_name]
        cond["n_tot"] = cond.loc[:,Elements_name].sum(axis = 1)
        evap = pd.concat(evap_list, axis = 1)[setting.total_isotopes_name + setting.Elements_name]
        X_Ni = X_grain[setting.isotopes_name["Ni"]].sum(axis = 1)
        gamma = activitity_coeff_Ni(X_Ni).values
        evap[setting.isotopes_name["Ni"] + ["Ni"]] = evap[setting.isotopes_name["Ni"] + ["Ni"]].mul(gamma, axis = 0)
        evap["n_tot"] = evap.loc[:,setting.Elements_name].sum(axis = 1)
        return cond, evap

    def condensation_and_evaporation_for_nucleus(self, T):
        """ This function would return the condenstion and evaporation rate for each isotopes"""
        # n_tot = self.df_nucleus["n_tot"]
        ## To save the computation drop the ((n_tot-1)/n_tot)**(2/3) for evaporation
        surface = self.df_nucleus["surface"]
        evap_list = []
        cond_list = []
        for metal in self.metal_list:
            name_list = setting.isotopes_name[metal.name]
            X_grain = self.df_nucleus[name_list].div(self.df_nucleus["n_tot"], axis = 0)
            P_evap = X_grain.mul(metal.p_sat)
            sqrt_mass = np.sqrt(2*math.pi*metal.mass_isotopes * setting.k * T)
            evap_rate = P_evap.div(sqrt_mass, axis = 1).mul(surface, axis = 0)
            if metal.name == "Ni":
                X_Ni = X_grain.sum(axis = 1)
                gamma = activitity_coeff_Ni(X_Ni.values)
                evap_rate = evap_rate.mul(gamma, axis = 0)
            evap_rate[metal.name] = evap_rate.sum(axis = 1)
            cond_rate = self.Gas.P[name_list].div(sqrt_mass, axis = 0).values[np.newaxis,:]\
                * surface[:,np.newaxis]
            cond_rate = pd.DataFrame(cond_rate, columns = name_list)
            cond_rate[metal.name] = cond_rate.sum(axis = 1)
            evap_list.append(evap_rate)
            cond_list.append(cond_rate)
        cond = pd.concat(cond_list, axis = 1)[setting.total_isotopes_name + setting.Elements_name]
        cond["n_tot"] = cond.loc[:,setting.Elements_name].sum(axis = 1)
        evap = pd.concat(evap_list, axis = 1)[setting.total_isotopes_name + setting.Elements_name]
        evap["n_tot"] = evap.loc[:,setting.Elements_name].sum(axis = 1)
        return cond, evap

    def Integrate_through_t(self, grad, dt):
        n_0 = self.df_nucleus["n_tot"]
        K = 3 * n_0.pow(1/3)
        net_grad = grad["n_tot"]
        growth = 1/27*(net_grad.div(n_0.pow(2/3), axis = 0) * dt + K).pow(3) - n_0
        return grad.div(net_grad, axis = 0).mul(growth, axis = 0)

    def update_nucleus_attribute(self, property_only = False):
        V = np.zeros(len(self.df_nucleus))
        if not property_only:
            n_tot = np.zeros(len(self.df_nucleus))
        for metal in self.metal_list:
            if not property_only:
                isotopes = setting.isotopes_name[metal.name]
                nucleus = self.df_nucleus[isotopes]
                self.df_nucleus[metal.name] = nucleus.apply(np.sum, axis = 1)
                n_tot += self.df_nucleus[metal.name]
            V = V + metal.volume * self.df_nucleus[metal.name]
        if not property_only:
            self.df_nucleus["n_tot"] = n_tot
        self.df_nucleus["volume"] = V
        self.df_nucleus["radius"] = (3/4/math.pi* self.df_nucleus["volume"]).pow(1/3)
        self.df_nucleus["surface"] = 4*math.pi* self.df_nucleus["radius"].pow(2)
        self.update_flag = False

    def run(self, J, dt):
        self.predict_steady_state_X_isotopes(self.main.T)
        growth = self.grow(dt)
        self.Jdt += J*dt
        if self.Jdt >= self.Jdt_threshold:
            self.generate_nucleus()
            self.update_nucleus_attribute(property_only = True)
            growth.loc[len(growth)] = self.df_nucleus.iloc[-1][growth.columns]
        else:
            self.update_nucleus_attribute(property_only = True)
        return growth.mul(self.df_nucleus["N"],axis = 0).sum(axis = 0)

    def run_normal(self, T, J, dt):
        cond, evap = self.condensation_and_evaporation_for_nucleus(self.main.T)
        grad = cond - evap
        growth = self.Integrate_through_t(grad, dt)
        self.df_nucleus[growth.columns] += growth
        self.update_flag = True
        self.Jdt += J*dt
        if self.Jdt >= self.Jdt_threshold:
            self.generate_nucleus()
            self.update_nucleus_attribute(property_only = True)
            growth.loc[len(growth)] = self.df_nucleus.iloc[-1][growth.columns]
        else:
            self.update_nucleus_attribute(property_only = True)
        return growth.mul(self.df_nucleus["N"],axis = 0).sum(axis = 0)

def steady_state_X_isotopes(X, cond, evap):
    x = X[:-1]
    K = X[-1]
    f = (cond.div(x, axis = 0).sub(evap) + K).values
    return np.append(f, np.sum(x) - 1)

def steady_state_X_isotopes_prime(X, cond, evap):
    x = X[:-1]
    fp = -cond.div(x**2, axis = 0).values
    fp = np.append(fp, 0)
    fp = np.diag(fp)
    fp[-1,:-1] = 1
    fp[:-1,-1] = 1
    return fp

def func_K_steady_state(Ks, cond, evap):
    gamma, K = Ks
    x = x_gamma_K(cond, evap, gamma, K)
    f1 = activitity_coeff_Ni(x[setting.isotopes_name["Ni"]].sum()) - gamma
    x_tot = np.sum(x)
    f2 = x_tot - 1
    return np.array((f1, f2))

def activitity_coeff_Ni(x):
    return 0.3888*x + 0.6112

def activitity_coeff_Ni_prime(x):
    return 0.3888

def x_gamma_K(cond, evap, gamma, K):
    evap_gamma = evap.copy()
    evap_gamma[setting.isotopes_name["Ni"]] *= gamma
    return cond.div(evap_gamma + K, axis = 0)

def x_gamma_K_prime(cond, evap, gamma, K):
    evap_gamma = evap.copy()
    evap_gamma[setting.isotopes_name["Ni"]] *= gamma
    common = -cond.div((evap_gamma + K).pow(2))
    d_gamma = np.sum(common[setting.isotopes_name["Ni"]] * evap[setting.isotopes_name["Ni"]])
    return d_gamma, common


def func_K_steady_state_prime(Ks, cond, evap):
    gamma, K = Ks
    d_gamma, common = x_gamma_K_prime(cond, evap, gamma, K)
    d_K = np.sum(common)
    d_x_Ni_d_k = np.sum(common[setting.isotopes_name["Ni"]])
    d_f = np.zeros((2,2))
    ## Simiplify! be careful.
    d_f[0,0] = activitity_coeff_Ni_prime(0) * d_gamma - 1
    d_f[0,1] = activitity_coeff_Ni_prime(0) * d_x_Ni_d_k
    d_f[1,0] = d_gamma
    d_f[1,1] = d_K
    return d_f
