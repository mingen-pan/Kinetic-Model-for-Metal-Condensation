import setting, metal_module, surface_energy, nucleation
import pandas as pd
from gas_module import Gas_Module
from gas_equilibrium import Gas_Equilibrium_Module
from condensation import Nucleus_Growth_Module
from nucleation import Nucleation
import numpy as np
import math
from joblib import Parallel, delayed

class Main():
    def __init__(self, T, dt, beta):
        self.T = T
        Gas, Equilibrium, metal_list = initial_parameters(self)
        self.metal_list = metal_list
        self.Nucleus = Nucleus_Growth_Module(self)
        self.Gas = Gas
        self.Equilibrium = Equilibrium
        self.dt = dt
        self.beta = beta
        self.collider = Collision_engineer()
        self.nucleation = Nucleation(self)


        # store data
        self.J_list = []
        self.grain_0_list = []
        self.Nucleus_saver = []
        self.P_atom_list = []
        self.P_tot_list = []
        self.delta_list = []

    def train(self, epoch = 1000, newTrain = True):
        self.p_atom_start = np.exp(self.Gas.ln_P_element)
        self.p_tot_start = self.Equilibrium.P_tot.copy()
        for i, t in enumerate(np.arange(0, epoch * self.dt, self.dt)):
            #updare enviormental paramerers
            if i % 100 == 0:
                print("iter: ", i)
            self.Temperature_decay()
            self.Equilibrium.update_T()
            p_collied, ln_p = self.Equilibrium.get_pressure()
            self.Gas.update_pressure(p_collied, ln_p)
            self.Gas.update_from_P()
            #record gas data
            self.P_atom_list.append(np.exp(self.Gas.ln_P_element))
            self.P_tot_list.append(self.Equilibrium.P_tot.copy())

            #update the clusters
            J = self.nucleation.run()
            delta = self.Nucleus.run(J, self.dt)
            ## update all the parameter
        #     Gas.N_1 -= delta
            self.Equilibrium.delta_gas(delta)
            self.delta_list.append(delta.copy())
            if i % 100 == 0:
                self.Nucleus_saver.append((self.T, self.Nucleus.df_nucleus.copy()))
            self.J_list.append(J)
            if len(self.Nucleus.df_nucleus) >= 1:
                grain_0 = self.Nucleus.df_nucleus.iloc[0].copy()
                self.grain_0_list.append(grain_0)

        self.Nucleus_saver.append((self.T, self.Nucleus.df_nucleus.copy()))


    def Temperature_decay(self, isovalue = "P", update_Gas = False):
        T_old = self.T
        self.T -= self.dt/self.beta
        for metal in self.metal_list:
            metal.episilon(self.T)
        if isovalue == "P":
            # N_std = P_std/(k*T)
            # Gas is updated from Equilibrium
            if update_Gas:
                self.Gas.N_1 *= T_old/T
                self.Gas.update_from_N_1()
            self.Nucleus.df_nucleus["N"] *= T_old/self.T
        elif isovalue == "V":
            P_gas *= math.exp(-dt/beta)

    def print_remain_gas(self):
        print(self.T)
        for name in setting.Elements_name:
            print(name +" Remain: ", self.Equilibrium.P_tot[name]/self.p_tot_start[name])

    def print_saturation(self):
        for name, metal in self.metal_list.items():
            print(name + " Saturation: ", self.Gas.P[name]/metal.p_sat)



class Collision_engineer():
    def __init__(self):
        pass
    def z(self, B, A, T):
        #Calculate the frequncy of A hitting on a B
        cross_section = math.pi*(B.R + A.R)**2
        mass = (A.mass * B.mass)/(A.mass + B.mass)
        vel = math.sqrt(8*setting.k * T/(math.pi*mass))
        return cross_section * vel * A.N
    def Z(self, B, A, T):
        #the collision frequency of A and B in the system
        cross_section = math.pi*(B.R + A.R)**2
        mass = (A.mass * B.mass)/(A.mass + B.mass)
        vel = math.sqrt(8*setting.k*T/(math.pi*mass))
        return cross_section * vel * A.N * B.N_eq
    def z_for_nucleus(self, df_nucleus, cluster_groups_0, T):
        assert type(df_nucleus) == pd.core.frame.DataFrame
        if len(df_nucleus) == 0:
            return None
        rate_list = []
        for cluster in cluster_groups_0.clusters:
            idx = np.argwhere(cluster.n == 1)[0,0]
            metal = cluster.metal_list[idx]
            name = metal.name
            R_0 = metal.radius
            R_array = df_nucleus["radius"]
            cross_section = np.pi*(R_array + R_0)**2
            mass_0 = metal.mass_isotopes[np.newaxis, :]
            M_array = df_nucleus["mass"][:,np.newaxis]
            reduced_mass = (M_array * mass_0)/(M_array + mass_0)
            vel = np.sqrt(8*setting.k * T/(np.pi*reduced_mass))
            rate = cross_section[:,np.newaxis] * vel * cluster.N * metal.X_P[np.newaxis,:]
            rate_list.append(rate)
        return pd.DataFrame(np.concatenate(rate_list, axis = 1), columns = total_isotopes_name)



def initial_parameters(main, p_tot = 1e-4, dust_factor = 1e7):

    p_H = 2.43e10/2.431e10*2*p_tot*1e5
    p_O = 1.413e7/2.431e10*2*p_tot*1e5
    p_C = 7.079e6/2.431e10*2*p_tot*1e5
    p_C = p_O *0.5
    
    p_Fe = 8.380e5/2.431e10*2*p_tot*1e5 * dust_factor
    p_Ni = 4.780e4/2.431e10*2*p_tot*1e5 * dust_factor
    p_Ir = 0.6448/2.431e10*2*p_tot*1e5 * dust_factor
    p_Mo = 2.601/2.431e10*2*p_tot*1e5 * dust_factor
    p_Ru = 1.900/2.431e10*2*p_tot*1e5 * dust_factor
    p_Pt = 1.357/2.431e10*2*p_tot*1e5 * dust_factor
    p_Os = 0.6738/2.431e10*2*p_tot*1e5 * dust_factor
    p_W = 0.1277/2.431e10*2*p_tot*1e5 * dust_factor
    p_Re = 0.05509/2.431e10*2*p_tot*1e5 * dust_factor

    setting.setting(main.T) #Fe abundance

    X_P_Fe = np.array([0.0585, 0.9175, 0.0212, 0.0028])
    X_P_Ni = np.array([0.68077, 0.26223, 0.01140, 0.03635, 0.00926])
    X_P_Ir = np.array([0.373, 0.627])
    # X_P_Mo = np.array([1])
    X_P_Mo = np.array([0.14649, 0.09187, 0.15873, 0.16673, 0.09582, 0.24292, 0.09744])
    X_P_Ru = np.array([0.5, 0.5])
    X_P_Pt = np.array([0.5, 0.5])
    X_P_Os = np.array([0.5, 0.5])
    X_P_W = np.array([0.5, 0.5])
    X_P_Re = np.array([0.374, 0.626])

    mass_Fe = [53.9396,55.9349,56.9354,57.9333]
    mass_Ni = [57.9353429, 59.9307864, 60.9310560, 61.9283451, 63.9279660]
    mass_Ir = [190.9605940, 192.9629264]
    mass_Mo = [92,94,95,96,97,98, 100]
    mass_Ru = [101.07-1, 101.07+1]
    mass_Pt = [195.084-1, 195.084+1]
    mass_Os = [190.23-2, 190.23+2]
    mass_W = [183.84-2, 183.84+2]
    mass_Re = [184.9529550, 186.9557531]

    para_Fe = metal_module.acquire_para("thermodynamics.xlsx", 'Fe')
    para_Ni = metal_module.acquire_para("thermodynamics.xlsx", 'Ni')
    para_Ir = metal_module.acquire_para("thermodynamics.xlsx", 'Ir')
    para_Mo = metal_module.acquire_para("thermodynamics.xlsx", 'Mo')
    para_Ru = metal_module.acquire_para("thermodynamics.xlsx", 'Ru')
    para_Pt = metal_module.acquire_para("thermodynamics.xlsx", 'Pt')
    para_Os = metal_module.acquire_para("thermodynamics.xlsx", 'Os')
    para_W = metal_module.acquire_para("thermodynamics.xlsx", 'W')
    para_Re = metal_module.acquire_para("thermodynamics.xlsx", 'Re')

    Fe_sigma = surface_energy.surface_energy_calculator("Fe", 2.41, 2.123, 1811, 7.633, 7.09E-6)
    Ni_sigma = surface_energy.surface_energy_calculator("Ni", 2.37, 2.08, 1728, 9.927, 6.5888E-06)
    Ir_sigma = surface_energy.surface_energy_calculator("Ir", 3.00, 2.655, 2739, 10, 8.5203E-06)
    Mo_sigma = surface_energy.surface_energy_calculator("Mo", 3.00, 2.51, 2896, 12.424, 9.3340E-06)
    Ru_sigma = surface_energy.surface_energy_calculator("Ru", 3.00, 2.655, 2607, 10, 8.1706E-06)
    Pt_sigma = surface_energy.surface_energy_calculator("Pt", 2.48, 2.203, 2041, 10, 9.0948E-06)
    Os_sigma = surface_energy.surface_energy_calculator("Os", 3.45, 2.95, 3306, 10, 8.421E-06)
    W_sigma = surface_energy.surface_energy_calculator("W", 3.47, 2.765, 3695, 9.618, 9.5501E-06)
    Re_sigma = surface_energy.surface_energy_calculator("Re", 3.6, 3.133, 3458, 10, 8.8586E-06)

    Fe = metal_module.Metal("Fe",X_P_Fe, mass_Fe, 9.2732796e-26, 126e-12, Fe_sigma, 8.6e3, 2.4e-10, 0.89e13, para_Fe)
    Ni = metal_module.Metal("Ni",X_P_Ni, mass_Ni, 9.7462675e-26, 124e-12, Ni_sigma, 2.38e4, 2.15e-10, 0.83e13, para_Ni)
    Ir = metal_module.Metal("Ir",X_P_Ir, mass_Ir, 3.1918381e-25, 150e-12, Ir_sigma, 4.30e4, 2.72e-10, 0.84e13, para_Ir)
    Mo = metal_module.Metal("Mo",X_P_Mo, mass_Mo, 1.593121e-25, 155e-12, Mo_sigma, 4.86e4, 2.12e-10, 0.94e13, para_Mo)
    Ru = metal_module.Metal("Ru",X_P_Ru, mass_Ru, 1.6783067e-25, 148e-12, Ru_sigma, 3.83e4, 2.42e-10, 0.98e13, para_Ru)
    Pt = metal_module.Metal("Pt",X_P_Pt, mass_Pt, 3.2394457e-25, 153e-12, Pt_sigma, 3.66e4, 2.34e-10, 0.77e13, para_Pt)
    Os = metal_module.Metal("Os",X_P_Os, mass_Os, 3.1588432e-25, 149.5e-12, Os_sigma, 5e4, 2.7e-10, 0.92e13, para_Os)
    W = metal_module.Metal("W", X_P_W, mass_W, 3.0527348e-25, 155.7e-12, W_sigma, 5.82e4, 2.78e-10, 0.98e13, para_W)
    Re = metal_module.Metal("Re",X_P_Re, mass_Re, 3.0920397e-25, 152e-12, Re_sigma, 4.66e4, 2.74e-10, 1.01e13, para_Re)

    metal_list = pd.Series([Fe, Ni, Ir, Mo, Ru, Pt, Os, W, Re], setting.Elements_name)
    main.metal_list = metal_list
    Gas = Gas_Module(main, [p_Fe, p_Ni, p_Ir, p_Mo, p_Ru, p_Pt, p_Os, p_W, p_Re])
    main.Gas = Gas
    Equilibrium = Gas_Equilibrium_Module(main, [p_H, p_O, p_C], ["H", "O", "C"])
    main.Equilibrium = Equilibrium
    p_collied, ln_p = Equilibrium.get_pressure()
    Gas.update_pressure(p_collied, ln_p)
    Gas.update_from_P()

    for metal in metal_list:
        metal.episilon(main.T)

    return Gas, Equilibrium, metal_list
