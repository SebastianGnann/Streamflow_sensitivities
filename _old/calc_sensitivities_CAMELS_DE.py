import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import helper_fcts
from functions.sig_Sensitivity import sig_Sensitivity
from functions.sig_Sensitivity import sig_SensitivityBudyko
from old.sig_SensitivityWithStorage import sig_SensitivityWithStorage
import matplotlib as mpl
import functions.util_SnowModel as util_SnowModel
from mpl_toolkits.basemap import Basemap
import functions.util_TurcPike as util_TurcPike

mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

# to run TOSSH
import matlab.engine

eng = matlab.engine.start_matlab()

# prepare data
data_path = "D:/Data/"

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "figures/"
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
mean_P_list = []  # precip average
mean_PET_list = []  # PET average
mean_Q_list = []  # Q average
mean_T_list = []  # T average
cor_PET_P_list = []  # correlation between PET and P
sens_P_list = []  # precip sensitivity
sens_PET_list = []  # PET sensitivity
R2_list = []  # R2
sensitivity_len = []  # number of years used for sensitivity calculation
sens_P_alt1_list = []  # precip sensitivity
sens_PET_alt1_list = []  # PET sensitivity
R2_alt1_list = []  # R2
sens_P_alt2_list = []  # precip sensitivity
sens_PET_alt2_list = []  # PET sensitivity
R2_alt2_list = []  # R2
sens_P_Budyko_list = []  # precip sensitivity
sens_PET_Budyko_list = []  # PET sensitivity
R2_Budyko_list = []  # R2
sens_P_storage_list = []  # precip sensitivity
sens_PET_storage_list = []  # PET sensitivity
sens_Q_storage_list = []  # storage sensitivity
R2_storage_list = []  # R2
corr_P_PET_list = []  # correlation between P and PET
S_WB_list = []  # maximum storage derived from water balance
S_WB_annual_list = []  # maximum storage derived from water balance (annual)
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
    print(id)

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

    # todo: data checks: remove negative, incomplete, dodgy time series, NaNs, e.g. strange multi day cycles (reservoirs?)
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

    # todo: calculate seasonality

    # todo: calculate variance inflation fcator

    # Budyko / Turc as a comparison
    sens_P_Budyko, sens_PET_Budyko, R2_Budyko, _, _ , _  = sig_SensitivityBudyko(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, plot_results=False, use_delta=False, fit_intercept=False)

    # Python signatures - perhaps code in TOSSH
    # sensitivities
    # todo: update sensitivity code, e.g. seasonal sensitivities
    sens_P, sens_PET, R2, nr_years, _ , _  = sig_Sensitivity(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, plot_results=False)

    sens_P_alt1, sens_PET_alt1, R2_alt1, _, _ , _  = sig_Sensitivity(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, fit_intercept=True, plot_results=False)

    sens_P_alt2, sens_PET_alt2, R2_alt2, _, _ , _  = sig_Sensitivity(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=True, fit_intercept=False, plot_results=False)

    sens_P_alt3, sens_PET_alt3, R2_alt3, _, _ , _  = sig_Sensitivity(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=True, fit_intercept=True, plot_results=False)

    # Budyko / Turc as a comparison
    sens_P_Budyko, sens_PET_Budyko, R2_Budyko, _, _ , _  = sig_SensitivityBudyko(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, plot_results=False, use_delta=False, fit_intercept=False)

    sens_P_Budyko, sens_PET_Budyko, R2_Budyko, _, _ , _  = sig_SensitivityBudyko(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, plot_results=False, use_delta=True, fit_intercept=False)

    sens_P_Budyko, sens_PET_Budyko, R2_Budyko, _, _ , _  = sig_SensitivityBudyko(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, plot_results=False, use_delta=False, fit_intercept=True)

    sens_P_Budyko, sens_PET_Budyko, R2_Budyko, _, _ , _  = sig_SensitivityBudyko(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, plot_results=False, use_delta=True, fit_intercept=True)

    #dQ_dP, dQ_dPET = util_TurcPike.calculate_sensitivities(np.nanmean(df_tmp["precipitation_mean"]),
    #                                      np.nanmean(df_tmp["pet_hargreaves"]), 2)
    #print(np.round(dQ_dP,2), np.round(dQ_dPET,2))
    # strangely the 1st one fits best... but the rest is basically the same

    # with storage
    sens_P_storage, sens_PET_storage, sens_Q_storage, R2_storage, _ = sig_SensitivityWithStorage(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values)


    ''' 
    # storage
    # TODO: potentially get rid of that function and call Matlab directly
    S_WB, S_WB_annual, _, _, _ = sig_StorageWaterBalance(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, plot_results=False)

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
                                          nargout=1,  plot_results=False)

    TotalStorage = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_StorageTotal', eng,
                                          df_tmp["discharge_spec_obs"].values, df_tmp["date"].values,
                                          df_tmp["precipitation_mean"].values, df_tmp["pet_hargreaves"].values,
                                          df_tmp["temperature_mean"].values,
                                          nargout=1, plot_results=False)
    '''

    # append all calculated values
    gauge_id_native_list.append(id)
    gauge_id_list.append(gauge_id)
    frac_snow_list.append(frac_snow)
    mean_P_list.append(np.nanmean(df_tmp["precipitation_mean"]))
    mean_PET_list.append(np.nanmean(df_tmp["pet_hargreaves"]))
    mean_Q_list.append(np.nanmean(df_tmp["discharge_spec_obs"]))
    mean_T_list.append(np.nanmean(df_tmp["temperature_mean"]))
    cor_PET_P_list.append(df_tmp["precipitation_mean"].corr(df_tmp["pet_hargreaves"]))
    sens_P_list.append(sens_P)
    sens_PET_list.append(sens_PET)
    R2_list.append(R2)
    sensitivity_len.append(nr_years)
    sens_P_alt1_list.append(sens_P_alt1)
    sens_PET_alt1_list.append(sens_PET_alt1)
    R2_alt1_list.append(R2_alt1)
    sens_P_alt2_list.append(sens_P_alt2)
    sens_PET_alt2_list.append(sens_PET_alt2)
    R2_alt2_list.append(R2_alt2)
    sens_P_Budyko_list.append(sens_P_Budyko)
    sens_PET_Budyko_list.append(sens_PET_Budyko)
    R2_Budyko_list.append(R2_Budyko)
    sens_P_storage_list.append(sens_P_storage)
    sens_PET_storage_list.append(sens_PET_storage)
    sens_Q_storage_list.append(sens_Q_storage)
    R2_storage_list.append(R2_storage)
    corr_P_PET_list.append(df_tmp["precipitation_mean"].corr(df_tmp["pet_hargreaves"]))
    '''
    S_WB_list.append(S_WB)
    S_WB_annual_list.append(S_WB_annual)
    BFI5_list.append(BFI5)
    BFI90_list.append(BFI90)
    BFI_LH_list.append(BFI_LH)
    Recession_a_list.append(Recession_Parameters[0, 0])
    Recession_b_list.append(Recession_Parameters[0, 1])
    BaseflowRecessionK_list.append(BaseflowRecessionK)
    Storage_Baseflow_list.append(Storage_Baseflow)
    Hydraulic_Storage.append(HydraulicStorage)
    Total_Storage.append(TotalStorage)
    '''
    perc_complete_list.append(perc_complete)

eng.quit()

df = pd.DataFrame()
df["id"] = gauge_id_list  # check!
df["gauge_id"] = gauge_id_native_list
df["frac_snow_control"] = frac_snow_list
df["mean_P"] = mean_P_list
df["mean_PET"] = mean_PET_list
df["mean_Q"] = mean_Q_list
df["mean_T"] = mean_T_list
df["cor_PET_P"] = cor_PET_P_list
df["sens_P"] = sens_P_list
df["sens_PET"] = sens_PET_list
df["R2"] = R2_list
df["sensitivity_len"] = sensitivity_len
df["sens_P_alt1"] = sens_P_alt1_list
df["sens_PET_alt1"] = sens_PET_alt1_list
df["R2_alt1"] = R2_alt1_list
df["sens_P_alt2"] = sens_P_alt2_list
df["sens_PET_alt2"] = sens_PET_alt2_list
df["R2_alt2"] = R2_alt2_list
df["sens_P_Budyko"] = sens_P_Budyko_list
df["sens_PET_Budyko"] = sens_PET_Budyko_list
df["R2_Budyko"] = R2_Budyko_list
df["sens_P_storage"] = sens_P_storage_list
df["sens_PET_storage"] = sens_PET_storage_list
df["sens_Q_storage"] = sens_Q_storage_list
df["R2_storage"] = R2_storage_list
df["corr_P_PET"] = corr_P_PET_list
'''
df["S_WB"] = S_WB_list
df["S_WB_annual"] = S_WB_annual_list
df["BFI5"] = BFI5_list
df["BFI90"] = BFI90_list
df["BFI_LH"] = BFI_LH_list
df["Recession_a"] = Recession_a_list
df["Recession_b"] = Recession_b_list
df["BaseflowRecessionK"] = BaseflowRecessionK_list
df["Storage_Baseflow"] = Storage_Baseflow_list
df["Hydraulic_Storage"] = Hydraulic_Storage
df["Total_Storage"] = Total_Storage
'''
df["perc_complete"] = perc_complete_list
df["aridity_control"] = df["mean_PET"] / df["mean_P"]
df["runoff_ratio"] = df["mean_Q"] / df["mean_P"]

df = pd.merge(df_attr, df, on='gauge_id')

# quality control
# flags: level 1 = 1, level 2 = 2, did not pass checks = 0

# human impacts
df["humanimpact_flag"] = 0
df.loc[(df["artificial_surfaces_perc"] >= 0) & (df["artificial_surfaces_perc"] <= 10) & (
        df["dams_num"] < 1), "humanimpact_flag"] = 1
df.loc[(df["artificial_surfaces_perc"] > 10) & (df["artificial_surfaces_perc"] <= 20) & (
        df["dams_num"] < 1), "humanimpact_flag"] = 2
# land use change is currently not possible to check

# data quality
df["recordlength_flag"] = 0
df.loc[(df["record_length"] >= 40) & (df["perc_complete"] >= 0.9) & (
        df["data_gap"] < pd.Timedelta(days=1095)), "recordlength_flag"] = 1
df.loc[(df["record_length"] >= 20) & (df["record_length"] < 40) & (df["perc_complete"] >= 0.9) & (
        df["data_gap"] < pd.Timedelta(days=1095)), "recordlength_flag"] = 2
# NOTE: information on individual gauging stations, rating curve uncertainties, etc. not available

# other checks
df["dataquality_flag"] = 0
df.loc[(df["area_error"] < 0.1) & (df["mean_P"] > df["mean_Q"]), "dataquality_flag"] = 1
# check for strange Q values: nan, negative, many 0s, consecutive days with same flow, step changes -> ROBIN CHECKS?

# final data flag
df["data_flag"] = 0
df.loc[((df["humanimpact_flag"] == 1) | (df["humanimpact_flag"] == 2)) &
       ((df["recordlength_flag"] == 1) | (df["recordlength_flag"] == 2)) &
       ((df["dataquality_flag"] == 1) | (df["dataquality_flag"] == 2)), "data_flag"] = 2
df.loc[(df["humanimpact_flag"] == 1) & (df["recordlength_flag"] == 1) & (df["dataquality_flag"] == 1), "data_flag"] = 1

# create dataframe that only consists of "data_flag" not equal to 0

df.to_csv(results_path + 'camels_DE_sensitivities.csv', index=False)
print("Finished saving data.")

# load data
df = pd.read_csv(results_path + 'camels_DE_sensitivities.csv')

# plot results
#df_checked = df[df["data_flag"] != 0]
df = df[df["humanimpact_flag"] != 0]
df = df[df["perc_complete"] > 0.95]

# plot standard Budyko plot
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["aridity_control"], 1 - df["runoff_ratio"], s=10, c="tab:blue", alpha=0.8, lw=0)
axes.set_xlabel("Aridity [-]")
axes.set_ylabel("1 - Runoff ratio [-]")
axes.set_xlim([0, 2])
axes.set_ylim([-0.25, 1.25])
helper_fcts.plot_Budyko_limits(df["aridity_control"], 1 - df["runoff_ratio"], axes)
#helper_fcts.plot_Budyko_curve(np.linspace(0, 10, 100), axes)
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
axes.plot(E0_vec/P_vec, 1-util_TurcPike.calculate_streamflow(P_vec,E0_vec,2)/P_vec, c="black", linestyle="--")
fig.savefig(figures_path + "Budyko_plot_CAMELS_DE.png", dpi=600, bbox_inches='tight')
plt.close()

# plot map
fig, ax = plt.subplots(figsize=(4, 4))
m = Basemap(projection='robin', resolution='l', area_thresh=1000.0, lat_0=0, lon_0=0)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='lightgrey', lake_color='white')
m.drawmapboundary(fill_color='white')
x, y = m(df["gauge_lon"].values, df["gauge_lat"].values)
scatter = m.scatter(x, y, s=10, c=df["sens_P"], alpha=0.9, vmin=0.0, vmax=1, cmap='Blues')
# invert colormap
cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.3, aspect=20)
ax.set_xlim(np.min(x) * 0.99, np.max(x) * 1.01)
ax.set_ylim(np.min(y) * 0.99, np.max(y) * 1.01)
cbar.set_label('dQ/dP [-]', rotation=270, labelpad=15)
plt.tight_layout()
fig.savefig(figures_path + "sens_P_map_CAMELS_DE" + ".png", dpi=600, bbox_inches='tight')
plt.close()

# plot map
fig, ax = plt.subplots(figsize=(4, 4))
m = Basemap(projection='robin', resolution='l', area_thresh=1000.0, lat_0=0, lon_0=0)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='lightgrey', lake_color='white')
m.drawmapboundary(fill_color='white')
x, y = m(df["gauge_lon"].values, df["gauge_lat"].values)
scatter = m.scatter(x, y, s=10, c=df["sens_PET"], alpha=0.9, vmin=-1, vmax=0, cmap='Reds_r')  # invert colormap
cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.3, aspect=20)
ax.set_xlim(np.min(x) * 0.99, np.max(x) * 1.01)
ax.set_ylim(np.min(y) * 0.99, np.max(y) * 1.01)
cbar.set_label('dQ/dPET [-]', rotation=270, labelpad=15)
plt.tight_layout()
fig.savefig(figures_path + "sens_PET_map_CAMELS_DE" + ".png", dpi=600, bbox_inches='tight')
plt.close()

# plot sensitivity
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_P"], df["sens_PET"], s=10, c=df["cor_PET_P"], alpha=0.8, lw=0, vmin=-0.3, vmax=0.1)
axes.set_xlabel("dQ/dP [-]")
axes.set_ylabel("dQ/PET [-]")
axes.set_xlim([-0.5, 2.0])
axes.set_ylim([-1.5, 1.5])
#axes.plot([-2, 2], [2, -2], color='black', linestyle='--', linewidth=1)
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
dQdP, dQdE0 = util_TurcPike.calculate_sensitivities(P_vec,E0_vec,2)
axes.scatter(dQdP, dQdE0, s=10, c="lightgrey", alpha=0.8, lw=0)
plt.grid()
cbar = plt.colorbar(im, ax=axes)
fig.savefig(figures_path + "Sensitivity_plot_CAMELS_DE.png", dpi=600, bbox_inches='tight')

# plot sensitivity as function of aridity and compare to theoretical sensitivity
fig = plt.figure(figsize=(5, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["aridity_control"], df["sens_P"], s=10, c="tab:blue", alpha=0.8, lw=0, label="dQ/dP")
im = axes.scatter(df["aridity_control"], df["sens_PET"], s=10, c="tab:orange", alpha=0.8, lw=0, label="dQ/dPET")
im = axes.scatter(df["aridity_control"], df["sens_P_Budyko"], s=1, c="tab:blue", alpha=0.3, lw=0, label="dQ/dP Budyko")
im = axes.scatter(df["aridity_control"], df["sens_PET_Budyko"], s=1, c="tab:orange", alpha=0.3, lw=0, label="dQ/dPET Budyko")
axes.set_xlabel("Aridity [-]")
axes.set_ylabel("Q Sensitivity [-]")
axes.set_xlim([0, 2])
axes.set_ylim([-1.2, 1.2])
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
dQdP, dQdE0 = util_TurcPike.calculate_sensitivities(P_vec,E0_vec,2)
axes.plot(E0_vec/P_vec, dQdP, color='tab:blue', linestyle='--', linewidth=2)
axes.plot(E0_vec/P_vec, dQdE0, color='tab:orange', linestyle='--', linewidth=2)
axes.plot(E0_vec/P_vec, -dQdE0/dQdP, color='grey', linestyle='--', linewidth=2, label='-dQ/dPET/dQ/dP')
axes.legend()
plt.grid()
fig.savefig(figures_path + "Sensitivity_aridity_plot_CAMELS_DE.png", dpi=600, bbox_inches='tight')

# compare sens_P to sens_P_alt1 etc.
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3),constrained_layout=True)
im1=ax1.scatter(df["sens_P"],df["sens_P_alt1"],s=20,c=df["cor_PET_P"],alpha=0.8,lw=0,label="dQ/dP",vmin=-0.3,vmax=0.1)
ax1.set_xlabel("dQ/dP #1 [-]")
ax1.set_ylabel("dQ/dP #2 [-]")
ax1.set_xlim([-0.5, 2])
ax1.set_ylim([-0.5, 2])
ax1.grid()
im2=ax2.scatter(df["sens_PET"],df["sens_PET_alt1"],s=20,c=df["cor_PET_P"],alpha=0.8,lw=0,label="dQ/dPET",vmin=-0.3,vmax=0.1)
ax2.set_xlabel("dQ/dPET #1 [-]")
ax2.set_ylabel("dQ/dPET #2 [-]")
ax2.set_xlim([-2, 3])
ax2.set_ylim([-2, 3])
ax2.grid()
cbar = plt.colorbar(im2, ax=ax2, pad=0.25)
cbar.ax.set_title('Cor(P,PET)')
fig.savefig(figures_path + "Sensitivity_comparison_CAMELS_DE.png", dpi=600, bbox_inches='tight')

# compare sens_P to sens_P_alt1 etc.
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3),constrained_layout=True)
im1=ax1.scatter(df["sens_P"]*df["mean_Q"]/df["mean_P"],df["sens_P_alt2"]*df["mean_Q"]/df["mean_P"],s=20,c=df["cor_PET_P"],alpha=0.8,lw=0,label="dQ/dP",vmin=-0.5,vmax=0.1)
ax1.set_xlabel("dQ/dP [-]")
ax1.set_ylabel("dQ/dP alt2 [-]")
ax1.set_xlim([-0.5, 2])
ax1.set_ylim([-0.5, 2])
ax1.grid()
ax1.legend()
im2=ax2.scatter(df["sens_PET"]*df["mean_Q"]/df["mean_PET"],df["sens_PET_alt2"]*df["mean_Q"]/df["mean_PET"],s=20,c=df["cor_PET_P"],alpha=0.8,lw=0,label="dQ/dPET",vmin=-0.3,vmax=0.1)
ax2.set_xlabel("dQ/dPET [-]")
ax2.set_ylabel("dQ/dPET alt2 [-]")
ax2.set_xlim([-2, 2.5])
ax2.set_ylim([-2, 2.5])
ax2.grid()
ax2.legend()
plt.colorbar(im2, ax=ax2)
fig.savefig(figures_path + "Elasticity_comparison_CAMELS_DE.png", dpi=600, bbox_inches='tight')



# if Q is more controlled by P than by PET ... is the same true for ET or is there asymmetry?
# can we expain this with differnet storage pools?

# water year!