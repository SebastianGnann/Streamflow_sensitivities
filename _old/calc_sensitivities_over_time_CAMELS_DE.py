import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import helper_fcts
from functions.sig_Sensitivity import sig_Sensitivity
from functions.sig_Sensitivity import sig_SensitivityBudyko
import functions.util_SnowModel as util_SnowModel
from mpl_toolkits.basemap import Basemap
import functions.util_TurcPike as util_TurcPike
from functions.sig_Sensitivity import sig_SensitivityOverTime
from functions.util_Seasonality import calculate_seasonality_index
from functions.util_StepwiseRegression import stepwise_regression_plot

#mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

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
mean_P_list = []  # precip average
mean_PET_list = []  # PET average
mean_Q_list = []  # Q average
mean_T_list = []  # T average
cor_PET_P_list = []  # correlation between PET and P
sensitivity_len = []  # number of years used for sensitivity calculation
sens_P_list = []  # precip sensitivity
sens_PET_list = []  # PET sensitivity
R2_list = []  # R2
VIF_list = []  # variance inflation factor
sens_P_alt_list = []  # precip sensitivity
sens_PET_alt_list = []  # PET sensitivity
R2_alt_list = []  # R2
BFI5_list = []  # baseflow index with time window of 5 days
perc_complete_list = []  # percentage of complete data
record_length_list = []  # record length

df_temporal = pd.DataFrame()

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

    # remove NaNs at beginning
    first_valid_index = df_tmp["discharge_spec_obs"].first_valid_index()
    df_tmp = df_tmp.loc[first_valid_index:]

    # check data
    perc_complete = np.sum(~np.isnan(df_tmp["discharge_spec_obs"].values)) / len(df_tmp["discharge_spec_obs"].values)
    record_length = (df_tmp["date"].max() - df_tmp["date"].min()).days / 365.25

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

    # Budyko / Turc as a comparison
    sens_P_Budyko, sens_PET_Budyko, R2_Budyko, _, _, _  = sig_SensitivityBudyko(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, plot_results=False, use_delta=False, fit_intercept=False)

    # sensitivities
    sens_P, sens_PET, R2, nr_years, VIF, cor_PET_P = sig_Sensitivity(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=False, fit_intercept=False)
    sens_P_alt, sens_PET_alt, R2_alt, _, _, _ = sig_Sensitivity(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, use_delta=True, fit_intercept=True)

    #stepwise_regression_plot(Q=df_tmp["discharge_spec_obs"].values, P=df_tmp["precipitation_mean"].values,
    #                         PET=df_tmp["pet_hargreaves"].values, t=df_tmp["date"].values, plot_results=True, fit_intercept=False)

    # df_tmp.to_csv(results_path + "camels_DE_timeseries_" + str(id) + ".csv", index=False)

    # calculate sensitivities over time without intercept
    df_time = sig_SensitivityOverTime(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, id, plot_results=False, use_delta=False, window_years=20)
    df_time.insert(0, "gauge_id", id)

    # calculate sensitivities over time with intercept
    df_time_alt = sig_SensitivityOverTime(
        df_tmp["discharge_spec_obs"].values, df_tmp["date"].values, df_tmp["precipitation_mean"].values,
        df_tmp["pet_hargreaves"].values, id, plot_results=False, use_delta=True, window_years=20)
    df_time_alt.insert(0, "gauge_id", id)

    # add df_time_alt rows sens_P and sens_PET and R2 and VIF to df_time and name it all _alt
    df_time["sens_P_alt"] = df_time_alt["sens_P"]
    df_time["sens_PET_alt"] = df_time_alt["sens_PET"]
    df_time["R2_alt"] = df_time_alt["R2"]
    df_time["VIF_alt"] = df_time_alt["VIF"]

    # append df_time to df_temporal
    df_temporal = pd.concat([df_temporal, df_time], ignore_index=True)

    # calculate BFI
    BFI5 = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_BFI', eng,
                              df_tmp["discharge_spec_obs"].values, df_tmp["date"].values,
                              method='UKIH', parameters=[5], plot_results=False)

    # append all calculated values
    gauge_id_native_list.append(id)
    gauge_id_list.append(gauge_id)
    frac_snow_list.append(frac_snow)
    seasonality_index_list.append(seasonality_index)
    mean_P_list.append(np.nanmean(df_tmp["precipitation_mean"]))
    mean_PET_list.append(np.nanmean(df_tmp["pet_hargreaves"]))
    mean_Q_list.append(np.nanmean(df_tmp["discharge_spec_obs"]))
    mean_T_list.append(np.nanmean(df_tmp["temperature_mean"]))
    cor_PET_P_list.append(df_tmp["precipitation_mean"].corr(df_tmp["pet_hargreaves"]))
    sensitivity_len.append(nr_years)
    sens_P_list.append(sens_P)
    sens_PET_list.append(sens_PET)
    R2_list.append(R2)
    VIF_list.append(VIF)
    sens_P_alt_list.append(sens_P_alt)
    sens_PET_alt_list.append(sens_PET_alt)
    R2_alt_list.append(R2_alt)
    BFI5_list.append(BFI5)
    perc_complete_list.append(perc_complete)
    record_length_list.append(record_length)

eng.quit()

df = pd.DataFrame()
df["id"] = gauge_id_list  # check!
df["gauge_id"] = gauge_id_native_list
df["frac_snow_control"] = frac_snow_list
df["seasonality_index"] = seasonality_index_list
df["mean_P"] = mean_P_list
df["mean_PET"] = mean_PET_list
df["mean_Q"] = mean_Q_list
df["mean_T"] = mean_T_list
df["cor_PET_P"] = cor_PET_P_list
df["sensitivity_len"] = sensitivity_len
df["sens_P"] = sens_P_list
df["sens_PET"] = sens_PET_list
df["R2"] = R2_list
df["VIF"] = VIF_list
df["sens_P_alt"] = sens_P_alt_list
df["sens_PET_alt"] = sens_PET_alt_list
df["R2_alt"] = R2_alt_list
df["BFI5"] = BFI5_list
df["perc_complete"] = perc_complete_list
df["record_length"] = record_length_list
df["aridity_control"] = df["mean_PET"] / df["mean_P"]
df["runoff_ratio"] = df["mean_Q"] / df["mean_P"]

#df = pd.merge(df_attr, df, on='gauge_id')

df.to_csv(results_path + 'camels_DE_sensitivities_time.csv', index=False)
df_temporal.to_csv(results_path + 'camels_DE_temporal_sensitivities.csv', index=False)
print("Finished saving data.")

# load data
df = pd.read_csv(results_path + 'camels_DE_sensitivities_time.csv')
df_temporal = pd.read_csv(results_path + 'camels_DE_temporal_sensitivities.csv')

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
fig.savefig(figures_path + "sens_P_map_CAMELS_US" + ".png", dpi=600, bbox_inches='tight')
plt.close()