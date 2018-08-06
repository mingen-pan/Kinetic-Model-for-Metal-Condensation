import nucon, setting
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class DataProcess():
    def __init__(self, main):
        self.main = main
        self.df_nucleus = self.main.Nucleus.df_nucleus
        self.nucleus_X = self.df_nucleus.loc[:, setting.Elements_name + ['radius']]
        self.nucleus_X[setting.Elements_name] = \
            self.nucleus_X[setting.Elements_name].div(self.df_nucleus["n_tot"], axis = 0)


    def print_nucleus_density(self):
        print(self.df_nucleus["N"].sum())

    def plot_nucleus_data(self, variables, labels, factor = [1, 1]):
        plt.plot(self.df_nucleus[variables[0]] * factor[0],\
            self.df_nucleus[variables[1]] * factor[1], '.')
        if labels[0] is not None:
            plt.xlabel(labels[0])
        if labels[1] is not None:
            plt.ylabel(labels[1])

    def cumulative_fraction(self, element_names = setting.Elements_name):
        color_fraction(self.nucleus_X, self.nucleus_X["radius"], element_names)

    def isotopic_fraction_plot(self, name, isotopes, weights, unit_factor = 1, err_bar = False):
        if not err_bar:
            isotopic_ratio = 1.0/(weights[0] - weights[1])
            isotopic_ratio *= (self.df_nucleus[isotopes[0]]/self.df_nucleus[isotopes[1]]\
                /(self.main.p_tot_start[isotopes[0]]/self.main.p_tot_start[isotopes[1]])-1)*1e3
            plt.plot(self.df_nucleus["radius"] * unit_factor, isotopic_ratio, label = name)
        else:
            assert len(isotopes) == 3, "three isotopes are needed for err plot (you could put [a, b, b])"
            isotopic_ratio, err = error_bar_for_element(self.df_nucleus, isotopes, weights, std = self.main.p_tot_start)
            error_bar_plot(self.df_nucleus["radius"], isotopic_ratio, err, label = name)

    def show(self):
        plt.show()




def color_fraction(df, x, elements, normalized = False, alpha = 0.5):
	# colors = ["blue", "orange", "green", "red", "purple", "pink", "yellow", "brown", "gray", "olive", "cyan"]
	colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
	cumulate = df[elements].copy()
	if normalized:
		cumulate = cumulate.div(cumulate.sum(),axis = 0)
	prev = 0
	patch = []
	for i, element in enumerate(elements):
		cumulate[element] += prev
		# plt.plot(df_x, cumulate[element])
		plt.fill_between(1e9*x, prev, cumulate[element], interpolate=True, color = colors[i%len(colors)], alpha = alpha)
		patch.append(mpatches.Patch(color = colors[i%len(colors)], alpha = alpha, label= element))
		prev = cumulate[element]

	plt.xlabel("radius (nm)")
	plt.ylabel("cumulated fraction")
	plt.ylim([0,1])
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handles=patch)
	plt.show()


def error_bar_for_Ni(df, useful_yield = 0.03, sigma = 1, system_uncertainty = 2e-3):
	Ni_58 = df["Ni_58"] * useful_yield
	Ni_60 = df["Ni_60"] * useful_yield
	Ni_62 = df["Ni_62"] * useful_yield
	error_60_58 = np.sqrt(1/Ni_60.values + 1/Ni_58.values + system_uncertainty**2)
	error_60_58 = np.sqrt(1/Ni_62.values + 1/Ni_58.values + system_uncertainty**2)
	error = np.minimum(error_60_58/2, error_60_58/4)
	return sigma*error

def error_bar_for_Fe(df, useful_yield = 0.03, sigma = 1, system_uncertainty = 2e-3):
	Fe_54 = df["Fe_54"] * useful_yield
	Fe_56 = df["Fe_56"] * useful_yield
	Fe_57 = df["Fe_57"] * useful_yield
	error_56_54 = np.sqrt(1/Fe_54.values + 1/Fe_56.values + system_uncertainty**2)
	error_57_54 = np.sqrt(1/Fe_54.values + 1/Fe_57.values + system_uncertainty**2)
	error = np.minimum(error_56_54/2, error_57_54/3)
	return sigma * error

def error_bar_for_element(df, name, mass, std = None, useful_yield = 0.03, sigma = 1, system_uncertainty = 2e-3):
	R_0 = df[name[0]] * useful_yield
	R_1 = df[name[1]] * useful_yield
	R_2 = df[name[2]] * useful_yield
	error_01 = np.sqrt(1/R_0.values + 1/R_1.values + system_uncertainty**2)
	error_02 = np.sqrt(1/R_0.values + 1/R_2.values + system_uncertainty**2)
	error = np.minimum(error_01/abs(mass[0] - mass[1]), error_02/abs(mass[0] - mass[2]))
	if std is not None:
		ratio = (R_1/R_0/(std[name[1]]/std[name[0]]) - 1)
		return 1e3*ratio/abs(mass[0] - mass[1]), 1e3*sigma * error
	else:
		return 1e3*sigma * error

def error_bar_plot(x, y, error_bar, color = None, alpha = 0.5, label = None):
	# colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
	ub = y - error_bar
	lb = y + error_bar
	plt.plot(x, y, label = label)
	plt.fill_between(x, lb, ub, interpolate=True, alpha = alpha)
