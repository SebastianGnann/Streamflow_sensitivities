import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions.sig_Sensitivity import sig_Sensitivity
from functions.sig_Sensitivity import sig_SensitivityLog
from functions.sig_Sensitivity import sig_SensitivityAveraging
from functions.sig_Sensitivity import sig_SensitivityWithStorage
import matplotlib as mpl
import functions.util_SnowModel as util_SnowModel
from functions.util_Seasonality import calculate_seasonality_index
from mpl_toolkits.basemap import Basemap

mpl.use('TkAgg')  # or can use 'TkAgg'

# to run TOSSH
import matlab.engine

eng = matlab.engine.start_matlab()
from functions.matlab_helper import run_tossh_function

# prepare data
data_path = "D:/Data/"

# check if folders exist
results_path = "../results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "../figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

df_topo = pd.read_csv(data_path + "CAMELS_DE/CAMELS_DE_topographic_attributes.csv",
                      sep=',', skiprows=0, encoding='latin-1')
df_climate = pd.read_csv(data_path + "CAMELS_DE/CAMELS_DE_climatic_attributes.csv",
                         sep=',', skiprows=0, encoding='latin-1')
df_landcover = pd.read_csv(data_path + "CAMELS_DE/CAMELS_DE_landcover_attributes.csv",
                           sep=',', skiprows=0, encoding='latin-1')
df_humaninfluence = pd.read_csv(data_path + "CAMELS_DE/CAMELS_DE_humaninfluence_attributes.csv",
                                sep=',', skiprows=0, encoding='latin-1')

df_attr = pd.merge(df_topo, df_climate, on='gauge_id')
df_attr = pd.merge(df_attr, df_landcover, on='gauge_id')
df_attr = pd.merge(df_attr, df_humaninfluence, on='gauge_id')

gauge_id_native_list = []
gauge_id_list = []
frac_snow_list = []  # fraction of precipitation falling as snow
seasonality_index_list = []  # seasonality index
cor_PET_P_list = []  # correlation PET and P
mean_P_list = []  # precip average
mean_PET_list = []  # PET average
mean_Q_list = []  # Q average
mean_T_list = []  # T average
sens_P_list = []  # precip sensitivity
sens_PET_list = []  # PET sensitivity
R2_list = []  # R2
sens_P_alt_list = []  # precip sensitivity
sens_PET_alt_list = []  # PET sensitivity
R2_alt_list = []  # R2
sens_P_alt2_list = []  # precip sensitivity
sens_PET_alt2_list = []  # PET sensitivity
R2_alt2_list = []  # R2
sens_P_alt3_list = []  # precip sensitivity
sens_PET_alt3_list = []  # PET sensitivity
R2_alt3_list = []  # R2
sens_P_alt4_list = []  # precip sensitivity
sens_PET_alt4_list = []  # PET sensitivity
R2_alt4_list = []  # R2
sensitivity_len = []  # number of years used for sensitivity calculation
sens_P_storage_list = []  # precip sensitivity with storage
sens_PET_storage_list = []  # PET sensitivity with storage
sens_Q_storage_list = []  # storage sensitivity
R2_storage_list = []  # R2 with storage
sens_P_storage_alt_list = []  # precip sensitivity with storage
sens_PET_storage_alt_list = []  # PET sensitivity with storage
sens_Q_storage_alt_list = []  # storage sensitivity
R2_storage_alt_list = []  # R2 with storage
BFI5_list = []  # baseflow index with time window of 5 days
BFI90_list = []  # baseflow index with time window of 90 days
BFI_LH_list = []  # baseflow index with Lyne-Hollick filter
Recession_a_list = []  # recession parameter a
Recession_b_list = []  # recession parameter b
BaseflowRecessionK_list = []  # baseflow recession constant
Storage_Baseflow_list = []  # storage from baseflow
Hydraulic_Storage = []  # hydraulic storage
Total_Storage = []  # total storage
perc_complete_list = []  # percentage of complete data

for id in df_attr["gauge_id"]:
    # id = 'DE411290'
    # id = 'DE811960'
    # id = 'DE810570'

    print(id)
    # select name of the gauge based on id
    df_attr_tmp = df_attr[df_attr["gauge_id"] == id]

    gauge_id = f"camelsde_{id}"

    df_tmp = pd.read_csv(data_path + "CAMELS_DE/timeseries/CAMELS_DE_hydromet_timeseries_" + str(id) + ".csv", sep=',')
    df_tmp["date"] = pd.to_datetime(df_tmp["date"])
    df_sim = pd.read_csv(data_path + "CAMELS_DE/timeseries_simulated/CAMELS_DE_discharge_sim_" + str(id) + ".csv",
                         sep=',')
    df_sim["date"] = pd.to_datetime(df_sim["date"])
    df_tmp = pd.merge(df_tmp, df_sim[["date", "pet_hargreaves"]], on='date')

    # remove NaNs at beginning
    first_valid_index = df_tmp["discharge_spec_obs"].first_valid_index()
    df_tmp = df_tmp.loc[first_valid_index:]

    perc_complete = np.sum(~np.isnan(df_tmp["discharge_spec_obs"].values)) / len(
        df_tmp["discharge_spec_obs"].values)

    # calculate SWE and snow fraction
    df_snow = util_SnowModel.calculate_swe(df_tmp["precipitation_mean"].values,
                                           df_tmp["date"].values,
                                           df_tmp["temperature_mean"].values,
                                           plot_results=False)
    df_tmp["melt"] = df_snow["melt"]
    df_tmp["rain"] = df_snow["rain"]
    df_tmp["swe"] = df_snow["swe"]

    frac_snow = np.nanmean(df_tmp["melt"]) / np.nanmean(df_tmp["precipitation_mean"])

    seasonality_index = calculate_seasonality_index(
        df_tmp["precipitation_mean"].values, df_tmp["pet_hargreaves"].values, df_tmp["date"].values)

    # sensitivities
    sens_P, sens_PET, R2, nr_years, VIF, cor_PET_P = sig_Sensitivity(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=False, fit_intercept=False)
    sens_P_alt, sens_PET_alt, R2_alt, _, _, _ = sig_Sensitivity(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=True, fit_intercept=True)
    sens_P_alt2, sens_PET_alt2, R2_alt2, _, _, _ = sig_SensitivityLog(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=False, fit_intercept=True)
    sens_P_alt3, sens_PET_alt3, R2_alt3, _, _, _ = sig_SensitivityAveraging(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=False, fit_intercept=False)
    sens_P_alt4, sens_PET_alt4, R2_alt4, _, _, _ = sig_SensitivityAveraging(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=True, fit_intercept=True)

    sens_P_storage, sens_PET_storage, sens_Q_storage, R2_storage, _ = sig_SensitivityWithStorage(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=False, fit_intercept=False)
    sens_P_storage_alt, sens_PET_storage_alt, sens_Q_storage_alt, R2_storage_alt, _ = sig_SensitivityWithStorage(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=True, fit_intercept=True)

    # TODO: add single regression as well?

    # from functions.util_StepwiseRegression import stepwise_regression_plot
    # stepwise_regression_plot(Q=df_tmp["discharge_spec_obs"].values, P=df_tmp["precipitation_mean"].values,
    #                         PET=df_tmp["pet_hargreaves"].values, t=df_tmp["date"].values, plot_results=True,
    #                         fit_intercept=True)

    # TOSSH signatures
    # baseflow
    BFI5 = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_BFI', eng,
                              df_tmp["discharge_spec_obs"].values, df_tmp["date"].values,
                              method='UKIH', parameters=[5], plot_results=False)
    BFI90 = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_BFI', eng,
                               df_tmp["discharge_spec_obs"].values, df_tmp["date"].values,
                               method='UKIH', parameters=[90], plot_results=False)
    BFI_LH = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_BFI', eng,
                                df_tmp["discharge_spec_obs"].values, df_tmp["date"].values,
                                method='Lyne_Hollick', parameters=[0.925, 3], plot_results=False)

    # BFI5_control = sig_BFI(df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, method='UKIH', parameters=[5])

    # storage from baseflow
    Storage_Baseflow = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_StorageFromBaseflow', eng,
                                          df_tmp["discharge_spec_obs"].values, df_tmp["date"].values,
                                          df_tmp["precipitation_mean"].values, df_tmp["pet_hargreaves"].values,
                                          nargout=1, plot_results=False)

    # recession analysis
    Recession_Parameters, _ = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_RecessionAnalysis', eng,
                                                 df_tmp["discharge_spec_obs"].values, df_tmp["date"].values,
                                                 nargout=2, fit_individual=False, plot_results=False)
    Recession_Parameters = np.array(Recession_Parameters)

    BaseflowRecessionK = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_BaseflowRecessionK', eng,
                                            df_tmp["discharge_spec_obs"].values, df_tmp["date"].values,
                                            nargout=1, recession_length=5, plot_results=False)

    HydraulicStorage = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_StorageHydraulic', eng,
                                          df_tmp["discharge_spec_obs"].values, df_tmp["date"].values,
                                          df_tmp["precipitation_mean"].values, df_tmp["pet_hargreaves"].values,
                                          df_tmp["temperature_mean"].values,
                                          nargout=1, plot_results=False)

    TotalStorage = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_StorageTotal', eng,
                                      df_tmp["discharge_spec_obs"].values, df_tmp["date"].values,
                                      df_tmp["precipitation_mean"].values, df_tmp["pet_hargreaves"].values,
                                      df_tmp["temperature_mean"].values,
                                      nargout=1, plot_results=False)

    # append all calculated values
    gauge_id_native_list.append(id)
    gauge_id_list.append(gauge_id)
    frac_snow_list.append(frac_snow)
    seasonality_index_list.append(seasonality_index)
    cor_PET_P_list.append(cor_PET_P)
    mean_P_list.append(np.nanmean(df_tmp["precipitation_mean"]))
    mean_PET_list.append(np.nanmean(df_tmp["pet_hargreaves"]))
    mean_Q_list.append(np.nanmean(df_tmp["discharge_spec_obs"]))
    mean_T_list.append(np.nanmean(df_tmp["temperature_mean"]))
    sens_P_list.append(sens_P)
    sens_PET_list.append(sens_PET)
    R2_list.append(R2)
    sens_P_alt_list.append(sens_P_alt)
    sens_PET_alt_list.append(sens_PET_alt)
    R2_alt_list.append(R2_alt)
    sens_P_alt2_list.append(sens_P_alt2)
    sens_PET_alt2_list.append(sens_PET_alt2)
    R2_alt2_list.append(R2_alt2)
    sens_P_alt3_list.append(sens_P_alt3)
    sens_PET_alt3_list.append(sens_PET_alt3)
    R2_alt3_list.append(R2_alt3)
    sens_P_alt4_list.append(sens_P_alt4)
    sens_PET_alt4_list.append(sens_PET_alt4)
    R2_alt4_list.append(R2_alt4)
    sensitivity_len.append(nr_years)
    sens_P_storage_list.append(sens_P_storage)
    sens_PET_storage_list.append(sens_PET_storage)
    sens_Q_storage_list.append(sens_Q_storage)
    R2_storage_list.append(R2_storage)
    sens_P_storage_alt_list.append(sens_P_storage_alt)
    sens_PET_storage_alt_list.append(sens_PET_storage_alt)
    sens_Q_storage_alt_list.append(sens_Q_storage_alt)
    R2_storage_alt_list.append(R2_storage_alt)
    BFI5_list.append(BFI5)
    BFI90_list.append(BFI90)
    BFI_LH_list.append(BFI_LH)
    Recession_a_list.append(Recession_Parameters[0, 0])
    Recession_b_list.append(Recession_Parameters[0, 1])
    BaseflowRecessionK_list.append(BaseflowRecessionK)
    Storage_Baseflow_list.append(Storage_Baseflow)
    Hydraulic_Storage.append(HydraulicStorage)
    Total_Storage.append(TotalStorage)
    perc_complete_list.append(perc_complete)

eng.quit()

df = pd.DataFrame()
df["id"] = gauge_id_native_list  # check!
df["frac_snow_control"] = frac_snow_list
df["seasonality_index"] = seasonality_index_list
df["cor_PET_P"] = cor_PET_P_list
df["gauge_id"] = gauge_id_list
df["mean_P"] = mean_P_list
df["mean_PET"] = mean_PET_list
df["mean_Q"] = mean_Q_list
df["mean_T"] = mean_T_list
df["sens_P"] = sens_P_list
df["sens_PET"] = sens_PET_list
df["R2"] = R2_list
df["sens_P_alt"] = sens_P_alt_list
df["sens_PET_alt"] = sens_PET_alt_list
df["R2_alt"] = R2_alt_list
df["sens_P_alt2"] = sens_P_alt2_list
df["sens_PET_alt2"] = sens_PET_alt2_list
df["R2_alt2"] = R2_alt2_list
df["sens_P_alt3"] = sens_P_alt3_list
df["sens_PET_alt3"] = sens_PET_alt3_list
df["R2_alt3"] = R2_alt3_list
df["sens_P_alt4"] = sens_P_alt4_list
df["sens_PET_alt4"] = sens_PET_alt4_list
df["R2_alt4"] = R2_alt4_list
df["sensitivity_len"] = sensitivity_len
df["sens_P_storage"] = sens_P_storage_list
df["sens_PET_storage"] = sens_PET_storage_list
df["sens_Q_storage"] = sens_Q_storage_list
df["R2_storage"] = R2_storage_list
df["sens_P_storage_alt"] = sens_P_storage_alt_list
df["sens_PET_storage_alt"] = sens_PET_storage_alt_list
df["sens_Q_storage_alt"] = sens_Q_storage_alt_list
df["R2_storage_alt"] = R2_storage_alt_list
df["BFI5"] = BFI5_list
df["BFI90"] = BFI90_list
df["BFI_LH"] = BFI_LH_list
df["Recession_a"] = Recession_a_list
df["Recession_b"] = Recession_b_list
df["BaseflowRecessionK"] = BaseflowRecessionK_list
df["Storage_Baseflow"] = Storage_Baseflow_list
df["Hydraulic_Storage"] = Hydraulic_Storage
df["Total_Storage"] = Total_Storage
df["perc_complete"] = perc_complete_list
df["aridity_control"] = df["mean_PET"] / df["mean_P"]

'''
# load attributes from Caravan
df_attributes_hydroatlas = pd.read_csv(
    "D:/Data/CAMELS_DE/Caravan_extension_DE/Caravan_extension_DE/attributes/camelsch/attributes_hydroatlas_camelsch.csv",
    sep=',')
df_attributes_caravan = pd.read_csv(
    "D:/Data/CAMELS_DE/Caravan_extension_DE/Caravan_extension_DE/attributes/camelsch/attributes_caravan_camelsch.csv",
    sep=',')
df = df.merge(df_attributes_hydroatlas, on='gauge_id', how='left')
# df = df.merge(df_attributes_caravan, on='gauge_id', how='left')
'''

# add lat lon and other metadata not contained in Caravan
df["gauge_lon"] = df_attr["gauge_lon"]
df["gauge_lat"] = df_attr["gauge_lat"]

# todo: compare Caravan to native attributes

# save results
df.to_csv(results_path + 'camels_DE_signatures.csv', index=False)
print("Finished saving data.")

df = pd.read_csv(results_path + 'camels_DE_signatures.csv')

# plot results
# scatter plot
fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
line_x = np.linspace(-10, 10000, 100)
# add line with y=0.065
line_y = np.ones_like(line_x) * 0.065
ax.plot(line_x, line_y, 'k--')
im = ax.scatter(df["ele_mt_smx"], df["Hydraulic_Storage"], s=20, alpha=0.9, c=df["slp_dg_sav"], cmap='viridis',
                vmin=0.0, vmax=250)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim([0., 5000])
ax.set_ylim([0, 1000])
plt.colorbar(im)
plt.show()

# aridity check
fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
line_x = np.linspace(-10, 10, 100)
line_y = line_x
ax.plot(line_x, line_y, 'k--')
im = ax.scatter(df["aridity_control"], 1 / (df["ari_ix_sav"] / 100), s=20, alpha=0.9)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim([0., 1])
ax.set_ylim([0., 1])
plt.show()

# map
fig, ax = plt.subplots(figsize=(4, 4))
m = Basemap(projection='robin', resolution='l', area_thresh=1000.0, lat_0=0, lon_0=0)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='lightgrey', lake_color='white')
m.drawmapboundary(fill_color='white')
x, y = m(df["gauge_lon"].values, df["gauge_lat"].values)
scatter = m.scatter(x, y, s=20, c=df["BFI5"], alpha=0.9, vmin=0.3, vmax=0.8, cmap='viridis')  # invert colormap
cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.3, aspect=20)
ax.set_xlim(np.min(x) * 0.99, np.max(x) * 1.01)
ax.set_ylim(np.min(y) * 0.99, np.max(y) * 1.01)
cbar.set_label('BFI [-]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

################

df_ROBIN = pd.read_csv("D:/Python/ROBIN_CAMELS_DE/results/camels_de_ROBIN.csv")
df = df[df["id"].isin(df_ROBIN["ID"].values)]

# plot sensitivity
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_P"], df["sens_PET"], s=25, c=df["BFI5"], alpha=0.8, lw=0, vmin=0.3, vmax=0.8)
axes.set_xlabel("dQ/dP [-]")
axes.set_ylabel("dQ/PET [-]")
axes.set_xlim([0, 1.5])
axes.set_ylim([-1.0, 0.5])
# axes.plot([-2, 2], [2, -2], color='black', linestyle='--', linewidth=1)
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
import functions.util_TurcPike as util_TurcPike

dQdP, dQdE0 = util_TurcPike.calculate_sensitivities(P_vec, E0_vec, 2)
axes.scatter(dQdP, dQdE0, marker='o', s=10, c='grey', alpha=0.8, lw=0)
plt.grid()
cbar = plt.colorbar(im, ax=axes)
cbar.set_label('BFI [-]', rotation=270, labelpad=15)
plt.show()
plt.savefig(figures_path + "sensitivity_scatter_CAMELS_DE.png", dpi=600, bbox_inches='tight')

# Create figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), constrained_layout=True)
im1 = ax1.scatter(df["aridity_control"], df["sens_P"], s=10, c=df["BFI5"], vmin=0.3, vmax=0.8)
ax1.set_ylabel("P Sensitivity [-]")
ax1.set_xlim([0.2, 1.6])
# ax1.set_xscale('log')
ax1.set_ylim([-0.5, 1.5])
im2 = ax2.scatter(df["aridity_control"], df["sens_PET"], s=10, c=df["BFI5"], vmin=0.3, vmax=0.8)
ax2.set_xlabel("Aridity [-]")
ax2.set_ylabel("PET Sensitivity [-]")
ax2.set_xlim([0.2, 1.6])
# ax2.set_xscale('log')
ax2.set_ylim([-1.5, 0.5])
# Common elements for both subplots
for ax in [ax1, ax2]:
    ax.plot([0, 20], [0, 0], color='grey', linestyle='--', linewidth=1)
    P_vec = np.linspace(0.01, 10, 100)
    E0_vec = np.linspace(10, 0.01, 100)
    dQdP, dQdE0 = util_TurcPike.calculate_sensitivities(P_vec, E0_vec, 2)
    Q = util_TurcPike.calculate_streamflow(P_vec, E0_vec, 2)
    if ax == ax1:
        ax.plot(E0_vec / P_vec, dQdP, color='grey', linestyle='--', linewidth=2)
    else:
        ax.plot(E0_vec / P_vec, dQdE0, color='grey', linestyle='--', linewidth=2)
cbar = fig.colorbar(im1, ax=[ax1, ax2], label='BFI [-]', aspect=30)
plt.savefig(figures_path + 'sensitivity_aridity_bfi_CAMELS_DE.png', dpi=300)

# Calculate deviations and correlations
df["dev_P"] = df["sens_P"] * df["mean_P"] / df["mean_Q"] - np.interp(1 / df["aridity_control"], P_vec / E0_vec,
                                                                     dQdP * P_vec / Q)
df["dev_PET"] = df["sens_PET"] * df["mean_PET"] / df["mean_Q"] - np.interp(1 / df["aridity_control"], P_vec / E0_vec,
                                                                           dQdE0 * E0_vec / Q)
# Calculate and print correlation between dev_P and BFI5
corr_P = df["dev_P"].corr(df["BFI5"])
corr_PET = df["dev_PET"].corr(df["BFI5"])
print("Correlation between deviation of P elasticity and BFI5: ", np.round(corr_P, 2))
print("Correlation between deviation of PET elasticity and BFI5: ", np.round(corr_PET, 2))

#### new checks
fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
im = ax.scatter(df_attr["p_seasonality"], df["seasonality_index"], s=20, alpha=0.9)
plt.show()
print(df["seasonality_index"].corr(df_attr["p_seasonality"], method='spearman'))
