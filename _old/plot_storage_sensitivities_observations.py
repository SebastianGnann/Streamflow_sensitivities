import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from functions import util_Turc
import functions.plotting_fcts as pf
#mpl.use('TkAgg')

# prepare data
data_path = "D:/Data/"

# check if folders exist
results_path = "../results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "../figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

# load data
df_CAMELS_US = pd.read_csv(results_path + 'CAMELS_US_sensitivities.csv')
df_CAMELS_US["country"] = 2
# take all catchments

df_CAMELS_GB = pd.read_csv(results_path + 'CAMELS_GB_sensitivities.csv')
df_CAMELS_GB["country"] = 3
# UKBN
df_UKBN = pd.read_csv("../results/UKBN_Station_List_vUKBN2.0_1.csv")
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

# storage sensitivity
# Create figure with two subplots stacked vertically
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 7), constrained_layout=True)
im1 = ax1.scatter(df["aridity_control"], df["sens_P_storage_mr1"], s=5, c=df["BFI"], vmin=0, vmax=1, cmap="magma")
ax1.set_ylabel("P Sensitivity [-]")
ax1.set_xlim([0.1, 10])
ax1.set_xscale('log')
ax1.set_ylim([-0.5, 2])
im2 = ax2.scatter(df["aridity_control"], df["sens_PET_storage_mr1"], s=5, c=df["BFI"], vmin=0, vmax=1, cmap="magma")
ax2.set_ylabel("PET Sensitivity [-]")
ax2.set_xlim([0.1, 10])
ax2.set_xscale('log')
ax2.set_ylim([-1.5, 1])
im3 = ax3.scatter(df["aridity_control"], df["sens_Q_storage_mr1"], s=5, c=df["BFI"], vmin=0, vmax=1, cmap="magma")
ax3.set_xlabel("Aridity [-]")
ax3.set_ylabel("Qlag1 Sensitivity [-]")
ax3.set_xlim([0.1, 10])
ax3.set_xscale('log')
ax3.set_ylim([-0.5, 1.5])
for ax in [ax1, ax2, ax3]:
    ax.plot([0.1, 20], [0, 0], color='grey', linestyle='--', linewidth=1)
    P_vec = np.linspace(0.01, 10, 100)
    E0_vec = np.linspace(10, 0.01, 100)
    dQdP, dQdE0 = util_Turc.calculate_sensitivities(P_vec, E0_vec, n)
    Q_vec = util_Turc.calculate_streamflow(P_vec, E0_vec, n)
    if ax == ax1:
        ax.plot(E0_vec/P_vec, dQdP, color='grey', linestyle='--', linewidth=2)
    elif ax == ax2:
        ax.plot(E0_vec/P_vec, dQdE0, color='grey', linestyle='--', linewidth=2)
cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], label='BFI [-]', aspect=30)
plt.savefig(figures_path + 'storage_sensitivity_aridity.png', dpi=600)

df["dev_P"] = df["sens_P_storage_mr1"] - util_Turc.calculate_sensitivities(df["mean_P"], df["mean_PET"], n)[0]
df["dev_PET"] = df["sens_PET_storage_mr1"] - util_Turc.calculate_sensitivities(df["mean_P"], df["mean_PET"], n)[1]
corr_P = df["dev_P"].corr(df["sens_Q_storage_mr1"], method='spearman')
corr_PET = df["dev_PET"].corr(df["sens_Q_storage_mr1"], method='spearman')
print("Correlation between deviation of P sensitivity and Qlag1 sensitivity: ", np.round(corr_P,2))
print("Correlation between deviation of PET sensitivity and Qlag1 sensitivity: ", np.round(corr_PET, 2))

corr_P = df["BFI"].corr(df["sens_P_storage_mr1"], method='spearman')
corr_PET = df["BFI"].corr(df["sens_PET_storage_mr1"], method='spearman')
corr_Q = df["BFI"].corr(df["sens_Q_storage_mr1"], method='spearman')
print("Correlation between P sensitivity and BFI: ", np.round(corr_P,2))
print("Correlation between PET sensitivity and BFI: ", np.round(corr_PET, 2))
print("Correlation between Qlag1 sensitivity and BFI: ", np.round(corr_Q,2))

# plot histogram of sensitivities
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(df["sens_P_storage_mr1"], bins=np.linspace(-1.5,1.5,100), alpha=0.7, color='tab:blue', label='dQ/dP')
ax.hist(df["sens_PET_storage_mr1"], bins=np.linspace(-1.5,1.5,100), alpha=0.7, color='tab:orange', label='dQ/dPET')
ax.hist(df["sens_Q_storage_mr1"], bins=np.linspace(-1.5,1.5,100), alpha=0.7, color='tab:purple', label='dQ/dQlag1')
ax.set_xlabel("Sensitivity [-]")
ax.set_ylabel("Count")
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([0, 125])
fig.tight_layout()
plt.savefig(figures_path + 'storage_sensitivity_histogram.png', dpi=600)

# compare R2
fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
line_x = np.linspace(0, 1, 100)
line_y = line_x
ax.plot(line_x, line_y, 'k--')
im = ax.scatter(df["R2_mr1"], df["R2_storage_mr1"], s=20, alpha=0.7)
ax.set_xlabel("R2")
ax.set_ylabel("R2 with storage")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.savefig(figures_path + 'storage_sensitivity_R2_comparison.png', dpi=600)
mean_R2 = df["R2_mr1"].mean()
mean_R2_alt = df["R2_storage_mr1"].mean()
print("Mean R2: ", np.round(mean_R2, 2))
print("Mean R2_alt: ", np.round(mean_R2_alt, 2))
