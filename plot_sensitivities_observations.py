import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from functions import util_Turc
import functions.plotting_fcts as pf
import matplotlib.ticker as ticker
def fmt_func(x, pos):
    return f"{round(x, 1):.1f}"
mpl.use('TkAgg')

# prepare data
data_path = "D:/Data/"

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

# load data
df_CAMELS_US = pd.read_csv(results_path + 'CAMELS_US_sensitivities.csv')
df_CAMELS_US["country"] = 2
# take all catchments

df_CAMELS_GB = pd.read_csv(results_path + 'CAMELS_GB_sensitivities.csv')
df_CAMELS_GB["country"] = 3
# UKBN
df_UKBN = pd.read_csv("./results/UKBN_Station_List_vUKBN2.0_1.csv")
df_CAMELS_GB = df_CAMELS_GB[df_CAMELS_GB["gauge_id_native"].isin(df_UKBN["Station"].values)]

df_CAMELS_DE = pd.read_csv(results_path + 'CAMELS_DE_sensitivities.csv')
df_CAMELS_DE["country"] = 4
# ROBIN catchments
df_ROBIN = pd.read_csv("D:/Python/ROBIN_CAMELS_DE/results/camels_de_ROBIN.csv")
df_CAMELS_DE = df_CAMELS_DE[df_CAMELS_DE["gauge_id_native"].isin(df_ROBIN["ID"].values)]

df_CAMELS_AUS = pd.read_csv(results_path + 'CAMELS_AUS_sensitivities.csv')
df_CAMELS_AUS["country"] = 5
df_humans = pd.read_csv("D:/Data/CAMELS_AUS_v2/04_attributes/04_attributes/CatchmentAttributes_04_AnthropogenicInfluences.csv", sep=',', skiprows=0, encoding='latin-1')
df_humans.rename(columns={'station_id': 'gauge_id_native'}, inplace=True)
df_CAMELS_AUS = pd.merge(df_CAMELS_AUS, df_humans, on='gauge_id_native')
df_CAMELS_AUS = df_CAMELS_AUS[df_CAMELS_AUS["river_di"]<0.2]

df = pd.concat([df_CAMELS_US, df_CAMELS_GB, df_CAMELS_DE, df_CAMELS_AUS], ignore_index=True)

# filter catchments
df = df[df["perc_complete"] > 0.95]
df = df[df["len_years"] > 30]
df = df[df["frac_snow_control"] < 0.2]
df = df[df["mean_P"] > df["mean_Q"]]
# remove catch
df = df.reset_index()

n = 2.5 # Turc-Pike parameter

# plot histogram with country
fig, ax = plt.subplots(figsize=(4, 3))
vc = df["country"].value_counts()
vc.index = vc.index.astype(int)
vc.sort_index(ascending=True).plot(kind='bar', ax=ax, color='tab:blue', alpha=0.7)
ax.set_xlabel(" ")
ax.set_ylabel("Count")
plt.tight_layout()
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['USA', 'GB', 'DE', 'AUS'])
plt.savefig(figures_path + 'countries_histogram.png', dpi=600)

# plot histogram of sensitivities
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(df["sens_P_mr1"], bins=np.linspace(-1.5,1.5,100), alpha=0.7, color='tab:blue', label='dQ/dP')
ax.hist(df["sens_PET_mr1"], bins=np.linspace(-1.5,1.5,100), alpha=0.7, color='tab:orange', label='dQ/dPET')
#ax.hist(df["sens_Q_storage"], bins=np.linspace(-1.5,1.5,100), alpha=0.7, color='tab:black', label='dQ/dQlag1')
ax.set_xlabel(r"$s_P$ / $s_{Ep}$ [-]")
ax.set_ylabel("Count")
ax.set_xlim([-1.5, 1.5])
#ax.set_ylim([0, 100])
fig.tight_layout()
plt.savefig(figures_path + 'sensitivity_histogram.png', dpi=600)

# print fraction of PET sensitivities larger than 0
print("Fraction of PET sensitivities larger than 0: ", np.round(np.sum(df["sens_PET_mr1"] > 0) / len(df), 2))
print("Fraction of PET sensitivities larger than 0: ", np.round(np.sum(df["sens_PET_mr2"] > 0) / len(df), 2))

# plot histogram of elasticities
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(df["sens_P_mr1"]*df["mean_P"]/df["mean_Q"], bins=np.linspace(-2,3,100), alpha=0.7, color='tab:blue', label='eP')
ax.hist(df["sens_PET_mr1"]*df["mean_PET"]/df["mean_Q"], bins=np.linspace(-2,3,100), alpha=0.7, color='tab:orange', label='ePET')
ax.set_xlabel(r"$e_P$ / $e_{Ep}$ [-]")
ax.set_ylabel("Count")
ax.set_xlim([-2, 3])
plt.savefig(figures_path + 'elasticity_histogram.png', dpi=600)

# compare R2
fig = plt.figure(figsize=(3, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["R2_mr1"], df["R2_mr2"], s=5, c="black", alpha=0.8)
axes.set_xlabel(r"$R^2$ Mult. Reg. #1 [-]")
axes.set_ylabel(r"$R^2$ Mult. Reg. #2 [-]")
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
axes.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1)
plt.savefig(figures_path + 'R2_comparison.png', dpi=600)
# print mean and median R2 for both
median_R2_mr1 = np.median(df["R2_mr1"])
median_R2_mr2 = np.median(df["R2_mr2"])
median_R2_log = np.nanmedian(df["R2_log"])
median_R2_storage_mr1 = np.nanmedian(df["R2_storage_mr1"])
median_R2_storage_mr2 = np.nanmedian(df["R2_storage_mr2"])
print("Median R2 Mult. Reg. #1: ", np.round(median_R2_mr1, 2))
print("Median R2 Mult. Reg. #2: ", np.round(median_R2_mr2, 2))
print("Median R2 Mult. Reg. Log: ", np.round(median_R2_log, 2))
print("Median R2 Storage Mult. Reg. #1: ", np.round(median_R2_storage_mr1, 2))
print("Median R2 Storage Mult. Reg. #2: ", np.round(median_R2_storage_mr2, 2))

# compare methods
fig = plt.figure(figsize=(3, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_P_mr1"], df["sens_P_mr2"], s=5, c="tab:blue", alpha=0.8, lw=0, vmin=-0.5, vmax=0.0, cmap='magma')
axes.set_xlabel(r"$s_P$ Mult. Reg. #1 [-]")
axes.set_ylabel(r"$s_P$ Mult. Reg. #2 [-]")
axes.set_xlim([-0.1, 1.4])
axes.set_ylim([-0.1, 1.4])
axes.plot([-1, 3], [-1, 3], color='grey', linestyle='--', linewidth=1)
#plt.grid()
#cbar = plt.colorbar(im, ax=axes)
#cbar.set_label('Cor(P,PET) [-]')
plt.savefig(figures_path + 'sensitivity_comparison_P.png', dpi=600)
cor_P = df["sens_P_mr1"].corr(df["sens_P_mr2"], method='spearman')
print("Correlation between dQ/dP #1 and dQ/dP #2: ", np.round(cor_P, 2))

# compare methods
fig = plt.figure(figsize=(3, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_PET_mr1"], df["sens_PET_mr2"], s=5, c="tab:orange", alpha=0.8, lw=0, vmin=-0.5, vmax=0.0, cmap='magma') #c=df["cor_PET_P"]
axes.set_xlabel(r"$s_{Ep}$ Mult. Reg. #1 [-]")
axes.set_ylabel(r"$s_{Ep}$ Mult. Reg. #2 [-]")
axes.set_xlim([-2, 3])
axes.set_ylim([-2, 3])
axes.plot([-2, 3], [-2, 3], color='grey', linestyle='--', linewidth=1)
#plt.grid()
#cbar = plt.colorbar(im, ax=axes)
#cbar.set_label('Cor(P,PET) [-]')
plt.savefig(figures_path + 'sensitivity_comparison_PET.png', dpi=600)
# print correlation
cor_PET = df["sens_PET_mr1"].corr(df["sens_PET_mr2"], method='spearman')
print("Correlation between dQ/dPET #1 and dQ/dPET #2: ", np.round(cor_PET, 2))

# plot sensitivity
import matplotlib.colors as mcolors
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
axes = plt.axes()
norm = mcolors.Normalize(vmin=0, vmax=2)
im = axes.scatter(df["sens_P_mr1"], df["sens_PET_mr1"], s=10, c=df["aridity_control"], alpha=0.8, lw=0, norm=norm, cmap="RdYlBu_r")
axes.set_xlabel(r"$s_P$ [-]")
axes.set_ylabel(r"$s_{Ep}$ [-]")
axes.set_xlim([0., 1.4])
axes.set_ylim([-1.0, 0.5])
axes.plot([-10, 10], [10, -10], color='grey', linestyle='--', linewidth=1)
P_vec = np.linspace(0.001, 100, 10000)
E0_vec = np.linspace(100, 0.001, 10000)
dQdP, dQdE0 = util_Turc.calculate_sensitivities(P_vec, E0_vec, n)
axes.scatter(dQdP, dQdE0, s=5, c="black", alpha=0.8, lw=0)
axes.scatter(dQdP, dQdE0, s=2, c=E0_vec/P_vec, alpha=0.8, lw=0, norm=norm, cmap="RdYlBu_r")
cbar = plt.colorbar(im, ax=axes, extend='max')
cbar.set_label(r"$E_p$/$P$")
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['0', '1', '> 2'])
plt.savefig(figures_path + 'sensitivity_dP_dPET.png', dpi=600)

# plot elasticity
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_P_mr1"]*df["mean_P"]/df["mean_Q"], df["sens_PET_mr1"]*df["mean_PET"]/df["mean_Q"], s=10, c=df["aridity_control"], alpha=0.8, lw=0, vmin=0, vmax=2, cmap="RdYlBu_r")
axes.set_xlabel(r"$e_P$ [-]")
axes.set_ylabel(r"$e_{Ep}$ [-]")
axes.set_xlim([0, 5])
axes.set_ylim([-4, 1])
axes.plot([-10, 10], [10, -10], color='grey', linestyle='--', linewidth=1)
P_vec = np.linspace(0.001, 100, 10000)
E0_vec = np.linspace(100, 0.001, 10000)
dQdP, dQdE0 = util_Turc.calculate_sensitivities(P_vec, E0_vec, n)
Q_vec = util_Turc.calculate_streamflow(P_vec,E0_vec,n)
axes.scatter(dQdP*P_vec/Q_vec, dQdE0*E0_vec/Q_vec, s=5, c="black", alpha=0.8, lw=0)
axes.scatter(dQdP*P_vec/Q_vec, dQdE0*E0_vec/Q_vec, s=2, c=E0_vec/P_vec, alpha=0.8, lw=0, vmin=0, vmax=2, cmap="RdYlBu_r")
#axes.plot([-1, 6], [2, -5], color='grey', linestyle='--', linewidth=1)
#plt.grid()
cbar = plt.colorbar(im, ax=axes, extend='max')
cbar.set_label(r"$E_p$/$P$")
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['0', '1', '> 2'])
plt.savefig(figures_path + 'elasticitiy_dP_dPET.png', dpi=600)

# plot sensitivity as function of aridity and compare to theoretical sensitivity
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True, sharey=False)
ax = ax1
ax.scatter(df["aridity_control"], df["sens_P_mr1"], s=5, c="tab:blue", alpha=0.8, lw=0, label="dQ/dP (data)")
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
dQdP, dQdE0 = util_Turc.calculate_sensitivities(P_vec, E0_vec, n)
ax.plot(E0_vec/P_vec, dQdP, color='white', linestyle='-', linewidth=3, label="dQ/dP (theory)")
ax.plot(E0_vec/P_vec, dQdP, color='tab:blue', linestyle='-', linewidth=2, label="dQ/dP (theory)")
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.set_ylabel(r"$s_P$ [-]")
ax.set_xscale('log')
ax.set_xlim([0.1, 10])
ax.set_ylim([-0.5, 1.5])
ax = ax2
ax.scatter(df["aridity_control"], df["sens_PET_mr1"], s=5, c="tab:orange", alpha=0.8, lw=0, label="dQ/dPET (data)")
ax.plot(E0_vec/P_vec, dQdE0, color='white', linestyle='-', linewidth=3, label="dQ/dPET (theory)")
ax.plot(E0_vec/P_vec, dQdE0, color='tab:orange', linestyle='-', linewidth=2, label="dQ/dPET (theory)")
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.set_xlabel(r"$E_p$/$P$ [-]")
ax.set_ylabel(r"$s_{Ep}$ [-]")
ax.set_xscale('log')
ax.set_xlim([0.1, 10])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_func))
ax.set_ylim([-1.5, 0.5])
fig.tight_layout()
plt.savefig(figures_path + 'sensitivity_aridity_panels.png', dpi=600)
# print number of catchments above and below axis limits
print("Number of catchments with dQ/dP > 1.5: ", np.sum(df["sens_P_mr1"] > 1.5))
print("Number of catchments with dQ/dP < -0.5: ", np.sum(df["sens_P_mr1"] < -0.5))
print("Number of catchments with dQ/dPET > 0.5: ", np.sum(df["sens_PET_mr1"] > 0.5))
print("Number of catchments with dQ/dPET < -1.5: ", np.sum(df["sens_PET_mr1"] < -1.5))

# plot elasticity as function of aridity and compare to theoretical sensitivity
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True, sharey=False)
ax = ax1
ax.scatter(df["aridity_control"], df["sens_P_mr1"]*df["mean_P"]/df["mean_Q"], s=5, c="tab:blue", alpha=0.8, lw=0, label="dQ/dP (data)")
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
dQdP, dQdE0 = util_Turc.calculate_sensitivities(P_vec, E0_vec, n)
Q_vec = util_Turc.calculate_streamflow(P_vec, E0_vec, n)
ax.plot(E0_vec/P_vec, dQdP*P_vec/Q_vec, color='white', linestyle='-', linewidth=3, label="dQ/dP (theory)")
ax.plot(E0_vec/P_vec, dQdP*P_vec/Q_vec, color='tab:blue', linestyle='-', linewidth=2, label="dQ/dP (theory)")
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.set_ylabel(r"$e_P$ [-]")
ax.set_xscale('log')
ax.set_xlim([0.1, 10])
ax.set_ylim([-1, 5])
ax = ax2
ax.scatter(df["aridity_control"], df["sens_PET_mr1"]*df["mean_PET"]/df["mean_Q"], s=5, c="tab:orange", alpha=0.8, lw=0, label="dQ/dPET (data)")
ax.plot(E0_vec/P_vec, dQdE0*E0_vec/Q_vec, color='white', linestyle='-', linewidth=3, label="dQ/dPET (theory)")
ax.plot(E0_vec/P_vec, dQdE0*E0_vec/Q_vec, color='tab:orange', linestyle='-', linewidth=2, label="dQ/dPET (theory)")
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.set_xlabel(r"$E_p$/$P$ [-]")
ax.set_ylabel(r"$e_{Ep}$ [-]")
ax.set_xscale('log')
ax.set_xlim([0.1, 10])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_func))
ax.set_ylim([-4, 1])
fig.tight_layout()
plt.savefig(figures_path + 'elasticity_aridity_panels.png', dpi=600)
# print number of catchments above and below axis limits
print("Number of catchments with dQ/dP > 5: ", np.sum(df["sens_P_mr1"]*df["mean_P"]/df["mean_Q"] > 5))
print("Number of catchments with dQ/dP < -1: ", np.sum(df["sens_P_mr1"]*df["mean_P"]/df["mean_Q"] < -1))
print("Number of catchments with dQ/dPET > 1: ", np.sum(df["sens_PET_mr1"]*df["mean_PET"]/df["mean_Q"] > 1))
print("Number of catchments with dQ/dPET < -4: ", np.sum(df["sens_PET_mr1"]*df["mean_PET"]/df["mean_Q"] < -4))

# sensitivity and different variables
pf.plot_sensitivity_aridity_variable(df, n, "BFI", 0, 1, "magma_r", figures_path)
pf.plot_sensitivity_aridity_variable(df, n, "frac_snow_control", 0, 0.2, "magma_r", figures_path)
pf.plot_sensitivity_aridity_variable(df, n, "P_seasonality_index", -1, 1, "magma_r", figures_path)
pf.plot_sensitivity_aridity_variable(df, n, "country", 2, 5, "magma_r", figures_path)
pf.plot_sensitivity_aridity_variable(df, n, "R2_mr1", 0, 1, "magma_r", figures_path)
pf.plot_sensitivity_aridity_variable(df, n, "cor_PET_P", -0.7, 0.1, "magma_r", figures_path)