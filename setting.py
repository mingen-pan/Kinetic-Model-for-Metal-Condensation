from my_imports import *


def setting(T, e_name = ["Fe","Ni", "Ir", "Mo", "Ru", "Pt", "Os", "W", "Re"]):
    global k, ln_P, N_1, Elements_name, h_bar, P_std, N_std, N_a, R, num_cores
    global isotopes_name, total_isotopes_name
    Fe_isotopes_name = ["Fe_54", "Fe_56", "Fe_57", "Fe_58"]
    Ni_isotopes_name = ["Ni_58", "Ni_60", "Ni_61", "Ni_62", "Ni_64"]
    Ir_isotopes_name = ["Ir_191", "Ir_193"]
    Mo_isotopes_name = ["Mo_92", "Mo_94", "Mo_95", "Mo_96", "Mo_97", "Mo_98", "Mo_100"]
    Ru_isotopes_name = ["Ru_100", "Ru_102"]
    Pt_isotopes_name = ["Pt_194", "Pt_196"]
    Os_isotopes_name = ["Os_188", "Os_192"]
    W_isotopes_name = ["W_182", "W_186"]
    Re_isotopes_name = ["Re_185", "Re_187"]
    isotopes_name = {"Fe": Fe_isotopes_name, "Ni": Ni_isotopes_name, "Ir": Ir_isotopes_name, "Mo": Mo_isotopes_name, \
                     "Ru": Ru_isotopes_name, "Pt": Pt_isotopes_name, "Os": Os_isotopes_name, "W": W_isotopes_name,\
                    "Re": Re_isotopes_name}
    total_isotopes_name = []
    for name in isotopes_name.values():
        total_isotopes_name += name
    k = 1.38e-23
    h_bar = 1.055e-34
    P_std = 1e5
    N_std = P_std/(k*T)
    N_a = 6.022e23
    R = 8.314
    num_cores = 4
    # num_cores = multiprocessing.cpu_count()
    Elements_name = e_name

setting(1700, e_name = ["Fe","Ni", "Ir", "Mo", "Ru", "Pt", "Os", "W", "Re"])
