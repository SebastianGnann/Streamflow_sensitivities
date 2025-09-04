import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from functions import util_Turc
from functions.calc_sensitivities import initialize_result_lists, append_results, calculate_metrics
from mpl_toolkits.basemap import Basemap
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

# list to store all attributes
result_lists = initialize_result_lists()

for id in df_attr["gauge_id"]:
    print(id)

    gauge_id = f"camelsde_{id}"

    df_tmp = pd.read_csv(data_path + "CAMELS_DE/timeseries/CAMELS_DE_hydromet_timeseries_" + str(id) + ".csv", sep=',')
    df_tmp["date"] = pd.to_datetime(df_tmp["date"])
    # could also load it directly from Caravan "streamflow"
    df_caravan = pd.read_csv(data_path + "caravan_de/timeseries/csv/camelsde/camelsde_" + str(id) + ".csv", sep=',')
    df_caravan["date"] = pd.to_datetime(df_caravan["date"])
    df_tmp = pd.merge(df_tmp, df_caravan[["date", "total_precipitation_sum",
                                          "potential_evaporation_sum_FAO_PENMAN_MONTEITH", "potential_evaporation_sum_ERA5_LAND",
                                          "temperature_2m_mean"]], on='date')
    df_sim = pd.read_csv(data_path + "CAMELS_DE/timeseries_simulated/CAMELS_DE_discharge_sim_" + str(id) + ".csv", sep=',')
    df_sim["date"] = pd.to_datetime(df_sim["date"])
    df_tmp = pd.merge(df_tmp, df_sim[["date", "pet_hargreaves"]], on='date')

    # rename variables
    df_tmp = df_tmp.rename(columns={"discharge_spec_obs": "Q",
                                    "precipitation_mean": "P",
                                    "pet_hargreaves": "PET",
                                    "temperature_mean": "T"})
    # CAMELS-DE: precipitation_mean pet_hargreaves temperature_mean
    # Caravan: total_precipitation_sum potential_evaporation_sum_FAO_PENMAN_MONTEITH temperature_2m_mean
    # plot time series of pet_hargreaves potential_evaporation_sum_FAO_PENMAN_MONTEITH potential_evaporation_sum_ERA5_LAND
    #df_tmp.plot(x='date', y=['pet_hargreaves', 'potential_evaporation_sum_FAO_PENMAN_MONTEITH'], figsize=(10, 5))
    #plt.show()

    # calculate signatures
    wy = 10  # define water year
    gauge_results = calculate_metrics(df_tmp, id, gauge_id, wy)  # calculate all values
    append_results(result_lists, gauge_results)  # append all calculated values

df = pd.DataFrame(result_lists)  # create final dataframe

# load attributes from Caravan
df_attributes_hydroatlas = pd.read_csv("D:/Data/caravan_de/attributes/camelsde/attributes_hydroatlas_camelsde.csv", sep=',')
df_attributes_caravan = pd.read_csv("D:/Data/caravan_de/attributes/camelsde/attributes_caravan_camelsde.csv", sep=',')
df_attributes_other = pd.read_csv("D:/Data/caravan_de/attributes/camelsde/attributes_other_camelsde.csv", sep=',')
df = df.merge(df_attributes_hydroatlas, on='gauge_id', how='left')
df = df.merge(df_attributes_caravan, on='gauge_id', how='left')
df = df.merge(df_attributes_other, on='gauge_id', how='left')

# save results
df.to_csv(results_path + 'camels_DE_sensitivities.csv', index=False)
#df.to_csv(results_path + 'camels_DE_sensitivities_Caravan.csv', index=False)
print("Finished saving data.")

df = pd.read_csv(results_path + 'camels_DE_sensitivities.csv')
#df = pd.read_csv(results_path + 'camels_DE_sensitivities_Caravan.csv')

# transform data
import re
df["start_wateryear"] = df["start_wateryear"].apply(lambda x: np.array([float(y) for y in re.findall(r"\d{4}", x)]))
df["end_wateryear"] = df["end_wateryear"].apply(lambda x: np.array([float(y) for y in re.findall(r"\d{4}", x)]))
df["mean_P_over_time"] = df["mean_P_over_time"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["mean_PET_over_time"] = df["mean_PET_over_time"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["mean_Q_over_time"] = df["mean_Q_over_time"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["aridity_over_time"] = df["aridity_over_time"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["cor_PET_P_over_time"] = df["cor_PET_P_over_time"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["sens_P_over_time_mr1"] = df["sens_P_over_time_mr1"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["sens_PET_over_time_mr1"] = df["sens_PET_over_time_mr1"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["R2_over_time_mr1"] = df["R2_over_time_mr1"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["sens_P_over_time_mr2"] = df["sens_P_over_time_mr2"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["sens_PET_over_time_mr2"] = df["sens_PET_over_time_mr2"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["R2_over_time_mr2"] = df["R2_over_time_mr2"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))

# plot results

# plot sensitivity
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_P_mr1"], df["sens_PET_mr1"], s=5, c=df["aridity_control"], alpha=0.8, lw=0, vmin=0.5, vmax=2)
axes.set_xlabel("dQ/dP [-]")
axes.set_ylabel("dQ/PET [-]")
axes.set_xlim([0., 1.5])
axes.set_ylim([-1.0, 1.0])
#axes.plot([-2, 2], [2, -2], color='black', linestyle='--', linewidth=1)
P_vec = np.linspace(0.01, 10, 1000)
E0_vec = np.linspace(10, 0.01, 1000)
dQdP, dQdE0 = util_Turc.calculate_sensitivities(P_vec,E0_vec,2)
axes.scatter(dQdP, dQdE0, s=7, c="white", alpha=0.8, lw=0)
axes.scatter(dQdP, dQdE0, s=5, c=E0_vec/P_vec, alpha=0.8, lw=0, vmin=0.5, vmax=2)
plt.grid()
cbar = plt.colorbar(im, ax=axes)
cbar.set_label('Aridity [-]')

# map
fig, ax = plt.subplots(figsize=(4, 4))
m = Basemap(projection='robin', resolution='l', area_thresh=1000.0, lat_0=0, lon_0=0)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='lightgrey', lake_color='white')
m.drawmapboundary(fill_color='white')
x, y = m(df["gauge_lon"].values, df["gauge_lat"].values)
scatter = m.scatter(x, y, s=20, c=df["sens_P_mr1"], alpha=0.9, vmin=0., vmax=1.0, cmap='viridis')  # invert colormap
cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.3, aspect=20)
ax.set_xlim(np.nanmin(x) * 0.99, np.nanmax(x) * 1.01)
ax.set_ylim(np.nanmin(y) * 0.99, np.nanmax(y) * 1.01)
cbar.set_label('dQ/dP [-]', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()

# plot over time
fig = plt.figure(figsize=(6, 3), constrained_layout=True)
axes = plt.axes()
for i in range(len(df)):
    if df["len_years"][i] < 30:
        continue
    else:
        im = axes.plot(df["start_wateryear"][i]+10, df["sens_P_over_time_mr1"][i], alpha=0.25, c='tab:blue', lw=0.5)
        im = axes.plot(df["start_wateryear"][i]+10, df["sens_PET_over_time_mr1"][i], alpha=0.25, c='tab:orange', lw=0.5)
        #im = axes.plot(df["start_wateryear"][i]+10, df["aridity_over_time"][i], alpha=0.25, c='tab:orange', lw=0.5)
axes.axhline(0, color='grey', linestyle='--', linewidth=1)
axes.set_xlabel("Year")
axes.set_ylabel("dQ/dP and dQ/PET [-]")
axes.set_ylim([-1.5, 1.5])
plt.show()