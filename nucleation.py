import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import setting



class Nucleation():
    def __init__(self, main, df = None):
        self.main = main
        self.metal_list = main.metal_list
        self.collider = main.collider
        self.Gas = main.Gas
        self.pure_groups = generate_pure_cluster_groups(100, self.metal_list, self.collider)
        if df is None:
            df = pd.DataFrame(None, index = self.metal_list.index, columns = ['mass', 'radius'])
            for metal in self.metal_list:
                name = metal.name
                df.loc[name, 'mass'] = metal.mass
                df.loc[name, 'radius'] = metal.radius
            df.loc['W','mass'] += 2.6566962e-26
            df.loc['W','radius'] += 60e-12
        self.collied_gas = Collided_Group(df, self.metal_list)
        self.J_max_metal_old = -1

    def run(self):
        self.collied_gas.update_from_pressure(self.Gas.P, self.main.T)
        self.pure_groups.apply(lambda x: x.update_equilibrium(self.collied_gas, self.Gas, self.main.T))
        fluxs = self.pure_groups.apply(lambda x: x.flux())
        J_max = fluxs.sort_values()[::-1]
        effective_flux = min(np.sum(fluxs >0), 2)
        if effective_flux  == 0:
            J = 0
        else:
            J_max_metal = J_max[:effective_flux].index.values
            if (J_max_metal != self.J_max_metal_old).any():
                self.cluster_groups = generate_cluster_groups(100, J_max_metal, self.Gas, self.metal_list, self.collider)
                self.J_max_metal_old = J_max_metal.copy()
            N_eq_list, D_up_list = update_equilibrium_state(self.cluster_groups, self.collied_gas, self.Gas, self.main.T)
            J = Flux_steady_state(N_eq_list, D_up_list)

        return J

    def update_collided_gas(self):
        for name in self.collied_gas.index:
            self.collied_gas[name].N = 1e5 * self.main.Equilibrium.P_tot[name] / (setting.k * self.main.T)


class Cluster():
    def __init__(self, n_array, metal_list):
        assert type(n_array) == np.ndarray
        self.metal_list = metal_list
        self.N = 0
        self.N_eq = 0
        self.n_tot = np.sum(n_array)
        self.n = n_array
        self.X = n_array/self.n_tot
        mass = 0
        for i, metal in enumerate(self.metal_list):
            mass += metal.mass * self.n[i]
        self.mass = mass
#         self.eta = metal.eta
        self.geometry()
        self.D_up = 0
        self.D_down = 0
    def geometry(self):
        V = 0
        for i, metal in enumerate(self.metal_list):
            V += 4/3*math.pi*(metal.radius**3) * self.n[i]
        self.V = V
        self.R = (V*3/4/math.pi)**(1/3)
        self.A = 4*math.pi*self.R**2
    def chemical_potential(self, T):
        mu = 0
        for i, metal in enumerate(self.metal_list):
            if self.X[i] == 0:
                continue
            epsilon = metal.epsl
            surface_atom = min(self.n_tot*metal.surface_ratio, self.n_tot**(2/3))
            mu_i = setting.k * T * (-math.log(metal.p_sat/setting.P_std)  \
            + metal.epsl * (self.n_tot**(1/3) - 1))\
            + metal.gamma * (surface_atom - metal.min_surface_ref)
            mu += mu_i * self.X[i]

        mask = (self.X != 0)
        mu += np.sum(setting.k * T * self.X[mask]*np.log(self.X[mask]))
        self.mu = mu
        return self.mu
    def density_eq(self, Gas, T):
        mu = self.chemical_potential(T)
        ln_n_eq = 0
        G = 0
        for i, metal in enumerate(self.metal_list):
            ln_n_eq += self.n[i] * Gas.ln_P_element[i]
            G += self.n[i] * metal.G_1_std
        ln_n_eq += (G - mu)/(setting.k * T)
        if ln_n_eq >200:
            ln_n_eq = 200
        self.N_eq =  math.exp(ln_n_eq) * setting.N_std
        if self.N_eq < 1e-100:
            self.N_eq = 0
        return self.N_eq

class Collided_Group():
    def __init__(self, df, metal_list):
        self._clusters = pd.Series(None,index = df.index)
        for name in self._clusters.index:
            n_array = pd.Series(0, metal_list.index)
            n_array[name] = 1
            self._clusters.loc[name] = Cluster(n_array.values, metal_list)
            self._clusters.loc[name].mass = df.loc[name, 'mass']
            self._clusters.loc[name].R = df.loc[name, 'radius']
            self._clusters.loc[name].A = 4 * math.pi * df.loc[name, 'radius']**2
            self._clusters.loc[name].V = 4/3 * math.pi * df.loc[name, 'radius']**3
            self._clusters.loc[name].N = 0
        self.loc = self._clusters.loc
        self.index = self._clusters.index

    def __getitem__(self, key):
        return self._clusters[key]

    def __setitem__(self, key, item):
        self._clusters[key] = item

    def update_from_pressure(self, p, T):
        for name in self._clusters.index:
            self._clusters[name].N = p[name] / (setting.k * T)
        # self._clusters.apply(lambda x: x.__dict__.__setitem__('N', p[x.index] / (setting.k * T)))


    def update_from_atm(self, p, T):
        for name in self._clusters.index:
            self._clusters[name].N = 1e5 * p[name] / (setting.k * T)
        # self._clusters.apply(lambda x: x.__dict__.__setitem__('N', 1e5 * p[x.index] / (setting.k * T)))

    def update_N(self, N):
        for name in self._clusters.index:
            self._clusters[name].N = N[name]
        # self._clusters.apply(lambda x: x.__dict__.__setitem__('N', N[x.index]))



class Cluster_Group():
    def __init__(self, n_tot, metal_name, metal_list, collider):
        self.clusters = []
        self.n_tot = n_tot
        self.metal_name = metal_name
        self.collider = collider
        if len(metal_name) == 1:
            idx = setting.Elements_name.index(metal_name[0])
            n_array = np.zeros(len(metal_list))
            n_array[idx] = n_tot
            cluster = Cluster(n_array, metal_list)
            self.clusters.append(cluster)
        elif len(metal_name) == 2:
            idx = []
            for name in metal_name:
                idx.append(setting.Elements_name.index(name))
            for i in range(n_tot + 1):
                n_array = np.zeros(len(metal_list))
                n_array[idx[0]] = i
                n_array[idx[1]] = n_tot - i
                cluster = Cluster(n_array, metal_list)
                self.clusters.append(cluster)

    def update_equilibrium(self, cluster_gas, Gas, T):
        D_up_list = []
        N_eq_list = []
        for cluster in self.clusters:
            cluster.D_up = 0
            N_eq = cluster.density_eq(Gas, T)
            for gas_particles in cluster_gas._clusters:
                cluster.D_up += self.collider.z(cluster, gas_particles, T)
            N_eq_list.append(N_eq)
            D_up_list.append(cluster.D_up)
        D_up_list = np.array(D_up_list)
        N_eq_list = np.array(N_eq_list)
        self.N_eq = np.sum(N_eq_list)
        self.Z_up = np.sum(D_up_list * N_eq_list)
        self.D_up = self.Z_up/(self.N_eq + 1e-105)




class Pure_Cluster_Group():
    def __init__(self, n_max, name, metal_list, collider):
        self.clusters = []
        self.n_max = n_max
        self.name = name
        self.collider = collider
        idx = setting.Elements_name.index(name)
        for i in range(1, n_max + 1):
            n_array = np.zeros(len(setting.Elements_name))
            n_array[idx] = i
            cluster = Cluster(n_array, metal_list)
            self.clusters.append(cluster)

    def update_equilibrium(self, cluster_gas, Gas, T):
        self.clusters[0].N_eq = Gas.N_1[self.name]
        self.clusters[0].N = Gas.N_1[self.name]
        N_eq_list = []
        D_up_list = []
        for cluster in self.clusters:
            N_eq = cluster.density_eq(Gas, T)
            cluster.D_up = self.collider.z(cluster, cluster_gas[self.name], T)
            N_eq_list.append(N_eq)
            D_up_list.append(cluster.D_up)
        D_up_list[0] /= 0.5
        self.D_up_array = np.array(D_up_list)
        self.N_eq_array = np.array(N_eq_list)

    def flux(self):
        if 0 in self.N_eq_array:
            return 0
        Z = 1/(self.N_eq_array * self.D_up_array)
        return 1/np.sum(Z)

def generate_pure_cluster_groups(n_max, metal_list, collider):
    pure_groups = []
    for name in metal_list.index:
        cg = Pure_Cluster_Group(n_max, name, metal_list, collider)
        pure_groups.append(cg)
    return pd.Series(pure_groups, index = metal_list.index)


def generate_cluster_groups(n_max, metal_name, Gas, metal_list, collider):
    N_1 = Gas.N_1[metal_name]
    cluster_groups = []
    for i in range(1, n_max+1):
        cl = Cluster_Group(i, metal_name, metal_list, collider)
        cluster_groups.append(cl)
    update_N_1(cluster_groups, N_1)
    return cluster_groups

def update_N_1(cluster_groups, N_1):
    for i, cluster in enumerate(cluster_groups[0].clusters):
        cluster.N = N_1[i]
        cluster.N_eq = N_1[i]

def update_equilibrium_state(cluster_groups, collied_gas, Gas, T):
    N_eq_list = []
    D_up_list = []
    for cluster_group in cluster_groups:
        cluster_group.update_equilibrium(collied_gas, Gas, T)
        N_eq_list.append(cluster_group.N_eq)
        D_up_list.append(cluster_group.D_up)
        #Just to save calculation times
        neq = cluster_group.N_eq
        if neq < 1e-100 or neq > 1e30:
            break
    cluster_groups[0].D_up /= 2
    D_up_list[0] /= 2
    return N_eq_list, D_up_list



def Flux_steady_state(N_eq_list, D_up_list):
    if 0 in N_eq_list:
        return 0
    N_eq_array = np.array(N_eq_list)
    D_up_array = np.array(D_up_list)
    Z = 1/(N_eq_array * D_up_array)
    return 1/np.sum(Z)


def N_real_calculation(cluster_groups, J):
    N = len(cluster_groups)
    cluster_groups[0].N = cluster_groups[0].N_eq
    density_list = [cluster_groups[0].N]
    for i in range(N - 1):
        if cluster_groups[i+1].N_eq == 0:
            break
        cluster_groups[i+1].D_down = cluster_groups[i].Z_up/cluster_groups[i+1].N_eq
        cluster_groups[i+1].N = (cluster_groups[i].D_up * cluster_groups[i].N - J)/cluster_groups[i+1].D_down
        if cluster_groups[i+1].N <0:
            cluster_groups[i+1].N = 0
            density_list.append(cluster_groups[i+1].N)
            break
        density_list.append(cluster_groups[i+1].N)
    return density_list

def pure_group_flux(cluster_group, Gas, T):
    cluster_group.update_equilibrium(Gas, T)
    return cluster_group.flux()

def evaporation_for_nucleus(df_nucleus, metal_list):
    n_tot = df_nucleus[total_isotopes_name].apply(np.sum ,axis = 1)
    X_grain = df_nucleus[total_isotopes_name].div(n_tot, axis = 0)
    surface = df_nucleus["surface"]*((n_tot-1)/n_tot)**2
    P_nx_list = []
    for metal in metal_list:
#         rhs = 2/3*metal.eta * metal.surface * n_tot**(-1/3)
#         P_n = metal.p_sat * np.exp(rhs)
        P_n = metal.p_sat
        name_list = isotopes_name[metal.name]
        P_nx = X_grain[name_list].mul(P_n, axis = 0)
        P_nx = P_nx.div(np.sqrt(2*math.pi*metal.mass_isotopes*setting.k*setting.T), axis = 1).mul(surface, axis = 0)
        P_nx_list.append(P_nx)
    return pd.concat(P_nx_list, axis = 1)[total_isotopes_name]

def update_nucleus_attribute(df_nucleus, metal_list):
    V = np.zeros(len(df_nucleus))
    M = np.zeros(len(df_nucleus))
    for metal in metal_list:
        isotopes = isotopes_name[metal.name]
        nucleus = df_nucleus[isotopes]
        V = V + metal.volume * nucleus.apply(np.sum, axis = 1)
        M = M + nucleus.apply(lambda x: np.sum(x * metal.mass_isotopes), axis = 1)
    df_nucleus["volume"] = V
    df_nucleus["radius"] = (3/4/math.pi*df_nucleus["volume"]).pow(1/3)
    df_nucleus["surface"] = 4*math.pi*df_nucleus["radius"].pow(2)
    df_nucleus["mass"] = M
    return df_nucleus
