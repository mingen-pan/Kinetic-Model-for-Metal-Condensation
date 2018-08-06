import setting

class surface_energy_calculator():
    def __init__(self, name, sigma_0, sigma_T_m, T_m, S_fus, molar_volume):
        self.name = name
        self.sigma_0 = sigma_0
        self.sigma_T_m = sigma_T_m
        self.T_m = T_m
        self.S_fus = S_fus
        self.molar_volume = molar_volume
        self.surface_per_mole = 1.612*(setting.N_a)**(1/3)*self.molar_volume**(2/3)
        #Integration parameters
        self.A = 0.5*4*setting.R/T_m
        self.B = 0.8*setting.R
        self.C = 0.5*(S_fus - self.B)/(0.5*T_m)
        self.D = (-S_fus + 1.6*setting.R)
        self.A /= self.surface_per_mole
        self.B /= self.surface_per_mole
        self.C /= self.surface_per_mole
        self.D /= self.surface_per_mole
        #forward parameters (forward: from 0 to T)
        self.sigma_1_f = self.sigma_0 - self.A * (0.2*self.T_m)**2
        self.sigma_2_f = self.sigma_1_f - self.B * (0.3*self.T_m)
        #backward parameters (forward: from 0 to T)
        self.sigma_2_b = self.C * ((T_m)**2 - (0.5*T_m)**2) + self.D * (T_m - 0.5*T_m) + self.sigma_T_m
        self.sigma_1_b = self.B * (0.5*T_m - 0.2*T_m) + self.sigma_2_b

    def forward(self, T):
        if T > 0.5*self.T_m:
            delta = self.C * (T**2 - (0.5*self.T_m)**2) + self.D * (T - 0.5*self.T_m)
            return self.sigma_2_f - delta
        elif T > 0.2*self.T_m:
            return self.sigma_1_f - self.B * (T - 0.2*self.T_m)
        else:
            return self.sigma_0 - self.A * T**2
    def backward(self, T):
        if T > 0.5*self.T_m:
            delta = self.C * (self.T_m**2 - T**2) + self.D * (self.T_m - T)
            return self.sigma_T_m + delta
        elif T > 0.2*self.T_m:
            return self.B * (0.5*self.T_m - T) + self.sigma_2_b
        else:
            delta = self.A * ((0.2*self.T_m)**2 - T**2)
            return self.sigma_1_b + delta
    def get_value(self, T):
        sigma_f = self.forward(T)
        sigma_b = self.backward(T)
        weight = T/self.T_m
        return weight * sigma_b + (1 - weight) * sigma_f
