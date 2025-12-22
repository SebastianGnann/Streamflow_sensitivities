import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from functions import util_Turc
import re
import functions.plotting_fcts as pf
import matplotlib.ticker as ticker
def fmt_func(x, pos):
    return f"{round(x, 1):.1f}"
#mpl.use('TkAgg')

# prepare data
data_path = "D:/Data/"

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

### forcing data comparison

df_CAMELS_DE = pd.read_csv(results_path + 'CAMELS_DE_sensitivities.csv')
df_CAMELS_DE["country"] = 4
# ROBIN catchments
df_ROBIN = pd.read_csv("D:/Python/ROBIN_CAMELS_DE/results/camels_de_ROBIN.csv")
df_CAMELS_DE = df_CAMELS_DE[df_CAMELS_DE["gauge_id_native"].isin(df_ROBIN["ID"].values)]

df_CAMELS_DE_Caravan = pd.read_csv(results_path + 'CAMELS_DE_sensitivities_Caravan.csv')
df_CAMELS_DE_Caravan["country"] = 44
# ROBIN catchments
df_ROBIN = pd.read_csv("D:/Python/ROBIN_CAMELS_DE/results/camels_de_ROBIN.csv")
df_CAMELS_DE_Caravan = df_CAMELS_DE_Caravan[df_CAMELS_DE_Caravan["gauge_id_native"].isin(df_ROBIN["ID"].values)]

df_CAMELS_AUS = pd.read_csv(results_path + 'CAMELS_AUS_sensitivities.csv')
df_CAMELS_AUS["country"] = 5
df_humans = pd.read_csv("D:/Data/CAMELS_AUS_v2/04_attributes/04_attributes/CatchmentAttributes_04_AnthropogenicInfluences.csv", sep=',', skiprows=0, encoding='latin-1')
df_humans.rename(columns={'station_id': 'gauge_id_native'}, inplace=True)
df_CAMELS_AUS = pd.merge(df_CAMELS_AUS, df_humans, on='gauge_id_native')
df_CAMELS_AUS = df_CAMELS_AUS[df_CAMELS_AUS["river_di"]<0.2]

df_CAMELS_AUS_Caravan = pd.read_csv(results_path + 'CAMELS_AUS_sensitivities_Caravan.csv')
df_CAMELS_AUS_Caravan ["country"] = 55
df_humans = pd.read_csv("D:/Data/CAMELS_AUS_v2/04_attributes/04_attributes/CatchmentAttributes_04_AnthropogenicInfluences.csv", sep=',', skiprows=0, encoding='latin-1')
df_humans.rename(columns={'station_id': 'gauge_id_native'}, inplace=True)
df_CAMELS_AUS_Caravan = pd.merge(df_CAMELS_AUS_Caravan , df_humans, on='gauge_id_native')
df_CAMELS_AUS_Caravan = df_CAMELS_AUS_Caravan [df_CAMELS_AUS_Caravan ["river_di"]<0.2]

"""
df_CAMELS_AUS_SILO = pd.read_csv(results_path + 'CAMELS_AUS_sensitivities_SILO.csv')
df_CAMELS_AUS_SILO ["country"] = 555
df_humans = pd.read_csv("D:/Data/CAMELS_AUS_v2/04_attributes/04_attributes/CatchmentAttributes_04_AnthropogenicInfluences.csv", sep=',', skiprows=0, encoding='latin-1')
df_humans.rename(columns={'station_id': 'gauge_id_native'}, inplace=True)
df_CAMELS_AUS_SILO = pd.merge(df_CAMELS_AUS_SILO , df_humans, on='gauge_id_native')
df_CAMELS_AUS_SILO = df_CAMELS_AUS_SILO [df_CAMELS_AUS_SILO ["river_di"]<0.2]
"""

# compare CAMELS_DE and CAMELS_DE_Caravan P and PET sensitivities with two scatter plots
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ax[0].scatter(df_CAMELS_DE["sens_P_mr1"], df_CAMELS_DE_Caravan["sens_P_mr1"], s=10, alpha=0.5, label='P sensitivity')
ax[0].plot([-2, 2], [-2, 2], color='grey', linestyle='--')
ax[1].scatter(df_CAMELS_DE["sens_PET_mr1"], df_CAMELS_DE_Caravan["sens_PET_mr1"], s=10, alpha=0.5, label='PET sensitivity')
ax[1].plot([-2, 2], [-2, 2], color='grey', linestyle='--')
ax[0].set_xlabel(r'$s_{P}$ CAMELS DE')
ax[0].set_ylabel(r'$s_{P}$ CAMELS DE Caravan')
ax[0].set_xlim([-0.2, 1.8])
ax[0].set_ylim([-0.2, 1.8])
ax[1].set_xlabel(r'$s_{Ep}$ CAMELS DE')
ax[1].set_ylabel(r'$s_{Ep}$ CAMELS DE Caravan')
ax[1].set_xlim([-1.5, 0.5])
ax[1].set_ylim([-1.5, 0.5])
plt.tight_layout()
#plt.show()
plt.savefig(figures_path + 'CAMELS_DE_Caravan_comparison_sensitivities.png', dpi=600)
plt.close()
print("Correlation P sensitivity:", np.round(df_CAMELS_DE_Caravan["sens_P_mr1"].corr(df_CAMELS_DE["sens_P_mr1"], method='spearman'),2))
print("Correlation PET sensitivity:", np.round(df_CAMELS_DE_Caravan["sens_PET_mr1"].corr(df_CAMELS_DE["sens_PET_mr1"], method='spearman'),2))
mean_error_P = np.mean(np.abs(df_CAMELS_DE_Caravan["sens_P_mr1"] - df_CAMELS_DE["sens_P_mr1"]))# / np.abs(df_CAMELS_DE["sens_P_mr1"]))
mean_error_PET = np.mean(np.abs(df_CAMELS_DE_Caravan["sens_PET_mr1"] - df_CAMELS_DE["sens_PET_mr1"]))# / np.abs(df_CAMELS_DE["sens_PET_mr1"]))
print("Mean error P sensitivity:", np.round(mean_error_P, 2))
print("Mean error PET sensitivity:", np.round(mean_error_PET, 2))

fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].scatter(df_CAMELS_DE["mean_P"], df_CAMELS_DE_Caravan["mean_P"], s=10, alpha=0.5, label='P sensitivity')
ax[0].plot([0, 6], [0, 6], color='grey', linestyle='--')
ax[1].scatter(df_CAMELS_DE["mean_PET"], df_CAMELS_DE_Caravan["mean_PET"], s=10, alpha=0.5, label='PET sensitivity')
ax[1].plot([0, 3], [0, 3], color='grey', linestyle='--')
ax[0].set_xlabel(r'$P$ CAMELS DE')
ax[0].set_ylabel(r'$P$ CAMELS DE Caravan')
ax[0].set_xlim([1, 5])
ax[0].set_ylim([1, 5])
ax[1].set_xlabel(r'${E_p}$ CAMELS DE')
ax[1].set_ylabel(r'${E_p}$ CAMELS DE Caravan')
ax[1].set_xlim([1, 2.5])
ax[1].set_ylim([1, 2.5])
plt.tight_layout()
#plt.show()
plt.savefig(figures_path + 'CAMELS_DE_Caravan_comparison_forcing.png', dpi=600)
plt.close()
print("Correlation P :", np.round(df_CAMELS_DE_Caravan["mean_P"].corr(df_CAMELS_DE["mean_P"], method='spearman'),2))
print("Correlation PET :", np.round(df_CAMELS_DE_Caravan["mean_PET"].corr(df_CAMELS_DE["mean_PET"], method='spearman'),2))
mean_error_P = np.mean(np.abs(df_CAMELS_DE_Caravan["mean_P"] - df_CAMELS_DE["mean_P"]))# / np.abs(df_CAMELS_DE["mean_P"]))
mean_error_PET = np.mean(np.abs(df_CAMELS_DE_Caravan["mean_PET"] - df_CAMELS_DE["mean_PET"]))# / np.abs(df_CAMELS_DE["mean_PET"]))
print("Mean error P:", np.round(mean_error_P, 2))
print("Mean error PET:", np.round(mean_error_PET, 2))

# compare CAMELS_AUS and CAMELS_AUS_Caravan P and PET sensitivities with two scatter plots
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ax[0].scatter(df_CAMELS_AUS["sens_P_mr1"], df_CAMELS_AUS_Caravan["sens_P_mr1"], s=10, alpha=0.5, label='P sensitivity')
ax[0].plot([-2, 2], [-2, 2], color='grey', linestyle='--')
ax[1].scatter(df_CAMELS_AUS["sens_PET_mr1"], df_CAMELS_AUS_Caravan["sens_PET_mr1"], s=10, alpha=0.5, label='PET sensitivity')
ax[1].plot([-2, 2], [-2, 2], color='grey', linestyle='--')
ax[0].set_xlabel(r'$s_{P}$ CAMELS AUS ')
ax[0].set_ylabel(r'$s_{P}$ CAMELS AUS Caravan ')
ax[0].set_xlim([-0.2, 1.8])
ax[0].set_ylim([-0.2, 1.8])
ax[1].set_xlabel(r'$s_{Ep}$ CAMELS AUS')
ax[1].set_ylabel(r'$s_{Ep}$ CAMELS AUS Caravan')
ax[1].set_xlim([-1.5, 0.5])
ax[1].set_ylim([-1.5, 0.5])
plt.tight_layout()
#plt.show()
plt.savefig(figures_path + 'CAMELS_AUS_Caravan_comparison_sensitivities.png', dpi=600)
plt.close()
print("Correlation P sensitivity:", np.round(df_CAMELS_AUS_Caravan["sens_P_mr1"].corr(df_CAMELS_AUS["sens_P_mr1"], method='spearman'),2))
print("Correlation PET sensitivity:", np.round(df_CAMELS_AUS_Caravan["sens_PET_mr1"].corr(df_CAMELS_AUS["sens_PET_mr1"], method='spearman'),2))
mean_error_P = np.mean(np.abs(df_CAMELS_AUS_Caravan["sens_P_mr1"] - df_CAMELS_AUS["sens_P_mr1"]))# / np.abs(df_CAMELS_AUS["sens_P_mr1"]))
mean_error_PET = np.mean(np.abs(df_CAMELS_AUS_Caravan["sens_PET_mr1"] - df_CAMELS_AUS["sens_PET_mr1"]))# / np.abs(df_CAMELS_AUS["sens_PET_mr1"]))
print("Mean error P sensitivity:", np.round(mean_error_P, 2))
print("Mean error PET sensitivity:", np.round(mean_error_PET, 2))

fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].scatter(df_CAMELS_AUS["mean_P"], df_CAMELS_AUS_Caravan["mean_P"], s=10, alpha=0.5, label='P sensitivity')
ax[0].plot([0, 8], [0, 8], color='grey', linestyle='--')
ax[1].scatter(df_CAMELS_AUS["mean_PET"], df_CAMELS_AUS_Caravan["mean_PET"], s=10, alpha=0.5, label='PET sensitivity')
ax[1].plot([0, 8], [0, 8], color='grey', linestyle='--')
ax[0].set_xlabel(r'$P$ CAMELS AUS')
ax[0].set_ylabel(r'$P$ CAMELS AUS Caravan')
ax[0].set_xlim([0, 6])
ax[0].set_ylim([0, 6])
ax[1].set_xlabel(r'${E_p}$ CAMELS AUS')
ax[1].set_ylabel(r'${E_p}$ CAMELS AUS Caravan')
ax[1].set_xlim([1, 6])
ax[1].set_ylim([1, 6])
plt.tight_layout()
#plt.show()
plt.savefig(figures_path + 'CAMELS_AUS_Caravan_comparison_forcing.png', dpi=600)
plt.close()
print("Correlation P :", np.round(df_CAMELS_AUS_Caravan["mean_P"].corr(df_CAMELS_AUS["mean_P"], method='spearman'),2))
print("Correlation PET :", np.round(df_CAMELS_AUS_Caravan["mean_PET"].corr(df_CAMELS_AUS["mean_PET"], method='spearman'),2))
mean_error_P = np.mean(np.abs(df_CAMELS_AUS_Caravan["mean_P"] - df_CAMELS_AUS["mean_P"]))# / np.abs(df_CAMELS_DE["mean_P"]))
mean_error_PET = np.mean(np.abs(df_CAMELS_AUS_Caravan["mean_PET"] - df_CAMELS_AUS["mean_PET"]))# / np.abs(df_CAMELS_DE["mean_PET"]))
print("Mean error P:", np.round(mean_error_P, 2))
print("Mean error PET:", np.round(mean_error_PET, 2))
