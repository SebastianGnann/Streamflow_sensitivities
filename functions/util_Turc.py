import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def fmt_func(x, pos):
    return f"{round(x, 1):.1f}"

# theoretical curves based on Turc-Mezentsev-Pike model based on the following paper:
# https://hess.copernicus.org/articles/23/2339/2019/#&gid=1&pid=1

# define functions
def calculate_streamflow(P, E0, n):
    Q = P - (P**-n + E0**-n)**(-1/n)
    return Q

def calculate_sensitivities(P, E0, n):
    dQ_dP = 1 - (1 + (P/E0)**n)**(-1/n-1)
    dQ_dE0 =  - (1 + (E0/P)**n)**(-1/n-1)
    return dQ_dP, dQ_dE0

def plot_Turc_curves():

    # plot figure showing Q/P, dQ/dP, dQ/dE0 all vs. aridity, defined as E0/P for a range of P and E0
    P_vec = np.linspace(0.01, 10, 100)
    E0_vec = np.linspace(10, 0.01, 100)
    dQdP, dQdE0 = calculate_sensitivities(P_vec, E0_vec, 2)
    Q_vec = calculate_streamflow(P_vec, E0_vec, 2)
    aridity = E0_vec / P_vec

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4, 4), constrained_layout=True)

    axes[0].plot(aridity, 1- Q_vec/P_vec, color='tab:purple', linewidth=2, label='Q')
    axes[0].set_ylabel(r"1-$Q$/$P$ [-]")
    # Common elements for both subplots
    axes[0].plot([0, 20], [0, 0], color='grey', linestyle='--', linewidth=1)
    axes[0].plot([1, 20], [1, 1], color='grey', linestyle='-', linewidth=1)
    x = np.logspace(-1, 0, 100)
    y = x
    axes[0].plot(x, y, color='grey', linestyle='-', linewidth=1)
    axes[0].set_xscale('log')
    axes[0].set_xlim([0.1, 10])
    axes[0].set_xticklabels([])

    axes[1].plot(aridity, dQdP, color='tab:blue', linestyle='-', linewidth=2, label='dQ/dP')
    axes[1].set_ylabel(r"$s_P$ [-]")
    axes[1].plot([0, 20], [0, 0], color='grey', linestyle='--', linewidth=1)
    axes[1].set_xscale('log')
    axes[1].set_xlim([0.1, 10])
    axes[1].set_xticklabels([])

    axes[2].plot(aridity, dQdE0, color='tab:orange', linestyle='-', linewidth=2, label='dQ/dE0')
    #axes[2].plot(aridity, dQdP + dQdE0, color='grey', linestyle='--', linewidth=2, label='Sum')
    axes[2].set_xlabel(r"$E_p$/$P$ [-]")
    axes[2].set_ylabel(r"$s_{Ep}$ [-]")
    axes[2].plot([0, 20], [0, 0], color='grey', linestyle='--', linewidth=1)
    axes[2].set_xscale('log')
    axes[2].set_xlim([0.1, 10])
    axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(fmt_func))
    #plt.show()

def plot_sensitivities():
    # plot figure showing Q/P, dQ/dP, dQ/dE0 all vs. aridity, defined as E0/P for a range of P and E0
    P_vec = np.linspace(0.01, 10, 100)
    E0_vec = np.linspace(10, 0.01, 100)
    dQdP, dQdE0 = calculate_sensitivities(P_vec,E0_vec,2)
    Q = calculate_streamflow(P_vec,E0_vec,2)

    # plot sensitivities
    fig = plt.figure(figsize=(5, 3), constrained_layout=True)
    axes = plt.axes()
    axes.plot(E0_vec/P_vec, dQdP, color='tab:blue', linestyle='--', linewidth=2, label='dQ/dP')
    axes.plot(E0_vec/P_vec, dQdE0, color='tab:orange', linestyle='--', linewidth=2, label='dQ/dPET')
    #axes.plot(E0_vec/P_vec, Q/P_vec, color='tab:blue', linestyle=':', linewidth=2, label='Q')
    #axes.plot(E0_vec/P_vec, 1-Q/P_vec, color='tab:orange', linestyle=':', linewidth=2, label='E')
    axes.plot(E0_vec/P_vec, dQdP+dQdE0, color='grey', linestyle='--', linewidth=2, label='Sum')
    axes.set_xlabel(r"$E_p$/$P$")
    axes.set_ylabel("Sensitivity [-]")
    axes.set_xlim([0.1, 10])
    axes.set_xscale('log')
    #axes.set_ylim([-1.2, 1.2])
    plt.legend()
    plt.show()

def plot_elasticities():
    # plot figure showing Q/P, dQ/dP, dQ/dE0 all vs. aridity, defined as E0/P for a range of P and E0
    P_vec = np.linspace(0.01, 10, 100)
    E0_vec = np.linspace(10, 0.01, 100)
    dQdP, dQdE0 = calculate_sensitivities(P_vec,E0_vec,2)
    Q = calculate_streamflow(P_vec,E0_vec,2)

    # plot elasticities
    fig = plt.figure(figsize=(5, 3), constrained_layout=True)
    axes = plt.axes()
    axes.plot(E0_vec/P_vec, dQdP*P_vec/Q, color='tab:blue', linestyle='--', linewidth=2, label='dQ/dP * P/Q')
    axes.plot(E0_vec/P_vec, dQdE0*E0_vec/Q, color='tab:orange', linestyle='--', linewidth=2, label='dQ/dPET * PET/Q')
    axes.plot(E0_vec/P_vec, (dQdP*P_vec/Q+dQdE0*E0_vec/Q), color='grey', linestyle='--', linewidth=2, label='Sum')
    axes.set_xlabel(r"$E_p$/$P$")
    axes.set_ylabel("Elasticitiy [-]")
    axes.set_xlim([0.1, 10])
    axes.set_xscale('log')
    #axes.set_ylim([-1.2, 1.2])
    plt.legend()
    plt.show()