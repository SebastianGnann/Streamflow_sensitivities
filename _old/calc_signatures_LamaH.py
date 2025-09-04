import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions.sig_Sensitivity import sig_Sensitivity
from functions.sig_Sensitivity import sig_SensitivityLog
from old.sig_SensitivityWithStorage import sig_SensitivityWithStorage
from functions.sig_StorageWaterBalance import sig_StorageWaterBalance
import matplotlib as mpl
import functions.util_SnowModel as util_SnowModel
from functions.util_Seasonality import calculate_seasonality_index
from mpl_toolkits.basemap import Basemap

mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

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

df_gauge = pd.read_csv(data_path + "2_LamaH-CE_daily/D_gauges/1_attributes/Gauge_attributes.csv",
                       sep=';', skiprows=0, encoding='latin-1')
df_attr = pd.read_csv(data_path + "2_LamaH-CE_daily/A_basins_total_upstrm/1_attributes/Catchment_attributes.csv",
                      sep=';', skiprows=0, encoding='latin-1')
df_attr = pd.merge(df_gauge, df_attr, on='ID')

gauge_id_native_list = []
gauge_id_list = []
frac_snow_list = []  # fraction of precipitation falling as snow
seasonality_index_list = []  # seasonality index
cor_PET_P_list = []  # correlation between P and PET
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
sensitivity_len = []  # number of years used for sensitivity calculation
sens_P_storage_list = []  # precip sensitivity with storage
sens_PET_storage_list = []  # PET sensitivity with storage
sens_Q_storage_list = []  # storage sensitivity
R2_storage_list = []  # R2 with storage
S_WB_list = []  # maximum storage derived from water balance
S_WB_annual_list = []  # maximum storage derived from water balance (annual)
BFI5_list = []  # baseflow index with time window of 5 days
BFI90_list = []  # baseflow index with time window of 90 days
BFI_LH_list = []  # baseflow index with Lyne-Hollick filter
Recession_a_list = []  # recession parameter a
Recession_b_list = []  # recession parameter b
BaseflowRecessionK_list = []  # baseflow recession constant
Storage_Baseflow_list = []  # storage from baseflow
perc_complete_list = []  # percentage of complete data

for id in df_attr["ID"]:
    # id = 407
    #id = 137 # lamah_204297
    print(id)

    id_long = df_attr.loc[df_attr["ID"] == id, "govnr"].values[0]
    gauge_id = f"lamah_{id_long}"

    df_tmp = pd.read_csv(data_path + "2_LamaH-CE_daily/F_hydrol_model/2_timeseries/" + "ID_" + str(id) + ".csv",
                         sep=';')
    df_tmp["date"] = pd.to_datetime(
        df_tmp[['YYYY', 'MM', 'DD']].rename(columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}))
    df_tmp.loc[df_tmp["Qobs"] < 0, "Qobs"] = np.nan  # nan is -999 in CAMELS US
    area = df_attr.loc[df_attr["ID"] == id, "area_calc"].values[0]
    df_tmp["Qobs"] = df_tmp["Qobs"] * 86.4 / area  # to mm/day

    # todo: data checks: remove negative, incomplete, dodgy time series, NaNs, 0 flows
    perc_complete = np.sum(~np.isnan(df_tmp["Qobs"].values)) / len(df_tmp["Qobs"].values)

    # calculate SWE and snow fraction
    df_snow = util_SnowModel.calculate_swe(df_tmp["P_A"].values,
                                           df_tmp["date"].values,
                                           df_tmp["T_A"].values,
                                           plot_results=False)
    df_tmp["melt"] = df_snow["melt"]
    df_tmp["rain"] = df_snow["rain"]
    df_tmp["swe"] = df_snow["swe"]

    frac_snow = np.nanmean(df_tmp["melt"]) / np.nanmean(df_tmp["P_A"])

    seasonality_index = calculate_seasonality_index(
        df_tmp["P_A"].values, df_tmp["PET_A"].values, df_tmp["date"].values)

    # Python signatures - perhaps code in TOSSH
    # sensitivities
    sens_P, sens_PET, R2, nr_years, VIF, cor_PET_P = sig_Sensitivity(
        df_tmp["Qobs"].values, df_tmp["date"].values, df_tmp["P_A"].values, df_tmp["PET_A"].values, plot_results=True)
    sens_P_alt, sens_PET_alt, R2_alt, _, _ , _  = sig_Sensitivity(
        df_tmp["Qobs"].values, df_tmp["date"].values, df_tmp["P_A"].values, df_tmp["PET_A"].values, use_delta=True, fit_intercept=True)
    sens_P_alt2, sens_PET_alt2, R2_alt2, _, _ , _  = sig_SensitivityLog(
        df_tmp["Qobs"].values, df_tmp["date"].values, df_tmp["P_A"].values, df_tmp["PET_A"].values, use_delta=False, fit_intercept=True)
    sens_P_storage, sens_PET_storage, sens_Q_storage, R2_storage, _ = sig_SensitivityWithStorage(
        df_tmp["Qobs"].values, df_tmp["date"].values, df_tmp["P_A"].values, df_tmp["PET_A"].values)

    from functions.util_StepwiseRegression import stepwise_regression_plot
    stepwise_regression_plot(Q=df_tmp["Qobs"].values, P=df_tmp["P_A"].values,
                             PET=df_tmp["PET_A"].values, t=df_tmp["date"].values, plot_results=True, fit_intercept=True)

    # storage
    S_WB, S_WB_annual, _, _, _ = sig_StorageWaterBalance(
        df_tmp["Qobs"].values, df_tmp["date"].values, df_tmp["P_A"].values, df_tmp["PET_A"].values, plot_results=False)

    # TOSSH signatures
    # baseflow
    BFI5 = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_BFI', eng,
                              df_tmp["Qobs"].values, df_tmp["date"].values,
                              method='UKIH', parameters=[5], plot_results=False)
    BFI90 = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_BFI', eng,
                               df_tmp["Qobs"].values, df_tmp["date"].values,
                               method='UKIH', parameters=[90], plot_results=False)
    BFI_LH = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_BFI', eng,
                                df_tmp["Qobs"].values, df_tmp["date"].values,
                                method='Lyne_Hollick', parameters=[0.925, 3], plot_results=False)

    # storage from baseflow
    Storage_Baseflow = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_StorageFromBaseflow', eng,
                                          df_tmp["Qobs"].values, df_tmp["date"].values,
                                          df_tmp["P_A"].values, df_tmp["PET_A"].values,
                                          nargout=1, plot_results=False)

    # recession analysis
    Recession_Parameters, _ = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_RecessionAnalysis', eng,
                                                 df_tmp["Qobs"].values, df_tmp["date"].values,
                                                 nargout=2, fit_individual=False, plot_results=False)
    Recession_Parameters = np.array(Recession_Parameters)

    BaseflowRecessionK = run_tossh_function(r'D:/Matlab/TOSSH/TOSSH_code', 'sig_BaseflowRecessionK', eng,
                                            df_tmp["Qobs"].values, df_tmp["date"].values,
                                            nargout=1, recession_length=5, plot_results=False)

    # append all calculated values
    gauge_id_native_list.append(id)
    gauge_id_list.append(gauge_id)
    frac_snow_list.append(frac_snow)
    seasonality_index_list.append(seasonality_index)
    cor_PET_P_list.append(cor_PET_P)
    mean_P_list.append(np.nanmean(df_tmp["P_A"]))
    mean_PET_list.append(np.nanmean(df_tmp["PET_A"]))
    mean_Q_list.append(np.nanmean(df_tmp["Qobs"]))
    mean_T_list.append(np.nanmean(df_tmp["T_A"]))
    sens_P_list.append(sens_P)
    sens_PET_list.append(sens_PET)
    R2_list.append(R2)
    sens_P_alt_list.append(sens_P_alt)
    sens_PET_alt_list.append(sens_PET_alt)
    R2_alt_list.append(R2_alt)
    sens_P_alt2_list.append(sens_P_alt2)
    sens_PET_alt2_list.append(sens_PET_alt2)
    R2_alt2_list.append(R2_alt2)
    sensitivity_len.append(nr_years)
    sens_P_storage_list.append(sens_P_storage)
    sens_PET_storage_list.append(sens_PET_storage)
    sens_Q_storage_list.append(sens_Q_storage)
    R2_storage_list.append(R2_storage)
    S_WB_list.append(S_WB)
    S_WB_annual_list.append(S_WB_annual)
    BFI5_list.append(BFI5)
    BFI90_list.append(BFI90)
    BFI_LH_list.append(BFI_LH)
    Recession_a_list.append(Recession_Parameters[0, 0])
    Recession_b_list.append(Recession_Parameters[0, 1])
    BaseflowRecessionK_list.append(BaseflowRecessionK)
    Storage_Baseflow_list.append(Storage_Baseflow)
    perc_complete_list.append(perc_complete)

eng.quit()

df = pd.DataFrame()
df["id"] = gauge_id_native_list
df["gauge_id"] = gauge_id_list
df["frac_snow_control"] = frac_snow_list
df["seasonality_index"] = seasonality_index_list
df["cor_PET_P"] = cor_PET_P_list
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
df["sensitivity_len"] = sensitivity_len
df["sens_P_storage"] = sens_P_storage_list
df["sens_PET_storage"] = sens_PET_storage_list
df["sens_Q_storage"] = sens_Q_storage_list
df["R2_storage"] = R2_storage_list
df["S_WB"] = S_WB_list
df["S_WB_annual"] = S_WB_annual_list
df["BFI5"] = BFI5_list
df["BFI90"] = BFI90_list
df["BFI_LH"] = BFI_LH_list
df["Recession_a"] = Recession_a_list
df["Recession_b"] = Recession_b_list
df["BaseflowRecessionK"] = BaseflowRecessionK_list
df["Storage_Baseflow"] = Storage_Baseflow_list
df["perc_complete"] = perc_complete_list
df["aridity_control"] = df["mean_PET"] / df["mean_P"]

# load attributes from Caravan
df_attributes_hydroatlas = pd.read_csv("D:/Data/Caravan-Jan25-csv/attributes/lamah/attributes_hydroatlas_lamah.csv",
                                       sep=',')
df_attributes_caravan = pd.read_csv("D:/Data/Caravan-Jan25-csv/attributes/lamah/attributes_caravan_lamah.csv", sep=',')
df = df.merge(df_attributes_hydroatlas, on='gauge_id', how='left')
# df = df.merge(df_attributes_caravan, on='gauge_id', how='left')

# add lat lon and other metadata not contained in Caravan
import geopandas as gpd
from pyproj import Transformer

transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
df["gauge_lon"], df["gauge_lat"] = transformer.transform(df_attr['lon'].values, df_attr['lat'].values)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['gauge_lon'], df['gauge_lat']),
    crs="EPSG:4326"  # Now in geographic coordinates (degrees)
)

# save results
df.to_csv(results_path + 'LamaH_signatures.csv', index=False)
print("Finished saving data.")

df = pd.read_csv(results_path + 'LamaH_signatures.csv')

# plot results

# aridity check
fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
line_x = np.linspace(-10, 10, 100)
line_y = line_x
ax.plot(line_x, line_y, 'k--')
im = ax.scatter(df["aridity_control"], 1 / (df["ari_ix_sav"] / 100), s=20, alpha=0.9)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim([0., 1.5])
ax.set_ylim([0., 1.5])
plt.show()

# map BFI
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

# plot sensitivity
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_P"], df["sens_PET"], s=10, c=df["ele_mt_smx"], alpha=0.8, lw=0, vmin=0, vmax=3000)
axes.set_xlabel("dQ/dP [-]")
axes.set_ylabel("dQ/PET [-]")
axes.set_xlim([-0.5, 2.0])
axes.set_ylim([-1.5, 1.5])
#axes.plot([-2, 2], [2, -2], color='black', linestyle='--', linewidth=1)
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
import functions.util_TurcPike as util_TurcPike
dQdP, dQdE0 = util_TurcPike.calculate_sensitivities(P_vec,E0_vec,2)
axes.scatter(dQdP, dQdE0, c=E0_vec/P_vec, alpha=0.8, lw=0, vmin=0.5, vmax=2)
plt.grid()
cbar = plt.colorbar(im, ax=axes)
plt.show()

# plot sensitivity as function of aridity and compare to theoretical sensitivity
fig = plt.figure(figsize=(5, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(1/df["aridity_control"], df["sens_P"]*df["mean_P"]/df["mean_Q"], s=10, c="tab:blue", alpha=0.8, lw=0, label="dQ/dP")
im = axes.scatter(1/df["aridity_control"], df["sens_PET"]*df["mean_PET"]/df["mean_Q"], s=10, c="tab:orange", alpha=0.8, lw=0, label="dQ/dPET")
axes.set_xlabel("Humidity [-]")
axes.set_ylabel("Elasticitiy [-]")
axes.set_xlim([0, 4])
#axes.set_ylim([-1.5, 1.5])
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
dQdP, dQdE0 = util_TurcPike.calculate_sensitivities(P_vec,E0_vec,2)
Q = util_TurcPike.calculate_streamflow(P_vec,E0_vec,2)
axes.plot(P_vec/E0_vec, dQdP*P_vec/Q, color='tab:blue', linestyle='--', linewidth=2)
axes.plot(P_vec/E0_vec, dQdE0*E0_vec/Q, color='tab:orange', linestyle='--', linewidth=2)
axes.legend()
plt.grid()
