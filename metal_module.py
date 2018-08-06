import pandas as pd
import math
import numpy as np
from scipy.optimize import curve_fit
import setting



class Metal():
    def __init__(self, name, X_P, isotopes, mass, radius, surface_energy, E_over_k, R_e, w, para):
        assert len(para) == 3
        assert type(X_P) == np.ndarray
        self.name = name
        self.mass = mass
        #X_P should be an array
        self.X_P = X_P
        self.isotopes = isotopes
        self.mass_isotopes = np.array(isotopes) * 1.660540199e-27
        self.radius = radius
        self.surface = 4*math.pi*radius**2
        self.volume = 4/3*math.pi*radius**3
        self.surface_energy = surface_energy
        self.E_over_k = E_over_k
        self.R_e = R_e
        self.w = w
        self.para = para
        #use the surface of a cluster to calculate the number of surface atom.
        #This number should be lower than the n_tot.
        self.surface_ratio = self.surface_energy.surface_per_mole/setting.N_a/self.surface
        self.min_surface_ref = min(self.surface_ratio, 1)
    def G_gas(self, T):
        A, B, C = self.para
        self.G_1_std = Gibbs_energy_fit(T, A, B, C)*1e3/setting.N_a
        return self.G_1_std
    def P_sat(self, T):
        self.p_sat = setting.P_std*math.exp(-self.G_gas(T)/(setting.k * T))
        return self.p_sat
    def Gamma(self, T):
        self.gamma = self.surface_energy.get_value(T)*self.surface
        return self.gamma
    def episilon(self, T):
        h_bar = 1.055e-34
        p_sat = self.P_sat(T)
        gamma = self.Gamma(T)
        kT = setting.k * T
        self.epsl = -(2**(2/3)-1)*gamma/kT- self.E_over_k/T + \
        math.log(1 - math.exp(-h_bar*self.w/kT))\
        + h_bar*self.w/(2*kT) \
        - math.log(self.mass * self.R_e**2 * kT/(2 * h_bar**2)) \
        - math.log(p_sat/kT*(self.mass * kT/(4*math.pi*h_bar**2))**(-3/2))
        self.epsl *= 1/(2**(1/3) - 1)
        return self.epsl


def Gibbs_energy_fit(x, a, b, c):
    return a + b*x + c*x*np.log(x)
def acquire_para(file, name):
    df = pd.read_excel(file, name)
    df = df[["T","G_f"]].dropna(0)
    para,_ = curve_fit(Gibbs_energy_fit, df["T"].values,df["G_f"].values)
    return para
