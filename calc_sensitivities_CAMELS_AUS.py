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

df_loc = pd.read_csv(
    "D:/Data/CAMELS_AUS_v2/02_location_boundary_area/02_location_boundary_area/location_boundary_area.csv",
    sep=',', skiprows=0, encoding='latin-1')
df_climate = pd.read_csv("D:/Data/CAMELS_AUS_v2/05_hydrometeorology/05_hydrometeorology/ClimaticIndices.csv",
                         sep=',', skiprows=0, encoding='latin-1')
df_geology = pd.read_csv("D:/Data/CAMELS_AUS_v2/04_attributes/04_attributes/CatchmentAttributes_01_Geology&Soils.csv",
                         sep=',', skiprows=0, encoding='latin-1')
df_topo = pd.read_csv("D:/Data/CAMELS_AUS_v2/04_attributes/04_attributes/CatchmentAttributes_02_Topography&Geometry.csv",
                      sep=',', skiprows=0, encoding='latin-1')
df_landcover = pd.read_csv(
    "D:/Data/CAMELS_AUS_v2/04_attributes/04_attributes/CatchmentAttributes_03_LandCover&Vegetation.csv",
    sep=',', skiprows=0, encoding='latin-1')
df_humans = pd.read_csv(
    "D:/Data/CAMELS_AUS_v2/04_attributes/04_attributes/CatchmentAttributes_04_AnthropogenicInfluences.csv",
    sep=',', skiprows=0, encoding='latin-1')
df_other = pd.read_csv("D:/Data/CAMELS_AUS_v2/04_attributes/04_attributes/CatchmentAttributes_05_Other.csv",
                       sep=',', skiprows=0, encoding='latin-1')
df_hydro = pd.read_csv("D:/Data/CAMELS_AUS_v2/03_streamflow/03_streamflow/streamflow_signatures.csv",
                       sep=',', skiprows=0, encoding='latin-1')

df_climate.rename(columns={'ID': 'station_id'}, inplace=True)
df_hydro.rename(columns={'ID': 'station_id'}, inplace=True)
df_attr = pd.merge(df_loc, df_climate, on='station_id')
df_attr = pd.merge(df_attr, df_geology, on='station_id')
df_attr = pd.merge(df_attr, df_topo, on='station_id')
df_attr = pd.merge(df_attr, df_landcover, on='station_id')
df_attr = pd.merge(df_attr, df_humans, on='station_id')
df_attr = pd.merge(df_attr, df_other, on='station_id')
# rename "CatchID" in df_hydro to "station_id" to match
df_hydro.rename(columns={'CatchID': 'station_id'}, inplace=True)
df_attr = pd.merge(df_attr, df_hydro, on='station_id')
df_attr.rename(columns={'station_id': 'gauge_id'}, inplace=True)

# read time series
#df_P = pd.read_csv("D:/Data/CAMELS_AUS_v2/05_hydrometeorology/05_hydrometeorology/01_precipitation_timeseries/precipitation_SILO.csv", sep=',')
df_P = pd.read_csv("D:/Data/CAMELS_AUS_v2/05_hydrometeorology/05_hydrometeorology/01_precipitation_timeseries/precipitation_AGCD.csv", sep=',')
df_PET = pd.read_csv("D:/Data/CAMELS_AUS_v2/05_hydrometeorology/05_hydrometeorology/02_EvaporativeDemand_timeseries/et_morton_wet_SILO.csv", sep=',')
#df_Tmin = pd.read_csv("D:/Data/CAMELS_AUS_v2/05_hydrometeorology/05_hydrometeorology/03_Other/SILO/tmin_SILO.csv", sep=',')
#df_Tmax = pd.read_csv("D:/Data/CAMELS_AUS_v2/05_hydrometeorology/05_hydrometeorology/03_Other/SILO/tmax_SILO.csv", sep=',')
df_Tmin = pd.read_csv("D:/Data/CAMELS_AUS_v2/05_hydrometeorology/05_hydrometeorology/03_Other/AGCD/tmin_AGCD.csv", sep=',')
df_Tmax = pd.read_csv("D:/Data/CAMELS_AUS_v2/05_hydrometeorology/05_hydrometeorology/03_Other/AGCD/tmax_AGCD.csv", sep=',')
df_Q = pd.read_csv("D:/Data/CAMELS_AUS_v2/03_streamflow/03_streamflow/streamflow_mmd.csv", sep=',')

# list to store all attributes
result_lists = initialize_result_lists()

for id in df_attr["gauge_id"]:
    print(id)

    gauge_id = f"camelsaus_{id}"

    for df in [df_Q, df_P, df_PET, df_Tmin, df_Tmax]:
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df_tmp = df_Q[['date']].copy()
    df_tmp = df_tmp.merge(df_P[['date', id]].rename(columns={id: 'P'}), on='date', how='left')
    df_tmp = df_tmp.merge(df_PET[['date', id]].rename(columns={id: 'PET'}), on='date', how='left')
    df_tmp = df_tmp.merge(df_Tmin[['date', id]].rename(columns={id: 'Tmin'}), on='date', how='left')
    df_tmp = df_tmp.merge(df_Tmax[['date', id]].rename(columns={id: 'Tmax'}), on='date', how='left')
    df_tmp = df_tmp.merge(df_Q[['date', id]].rename(columns={id: 'Q'}), on='date', how='left')
    df_tmp['T'] = (df_tmp['Tmin'] + df_tmp['Tmax']) / 2
    df_tmp = df_tmp.drop(columns=['Tmin', 'Tmax'])
    df_tmp = df_tmp.replace(-99.99000, np.nan)

    # rename variables
    df_tmp = df_tmp.rename(columns={"Q": "Q",
                                    "P": "P",
                                    "PET": "PET",
                                    "T": "T"})

    # calculate signatures
    wy = 7  # define water year
    gauge_results = calculate_metrics(df_tmp, id, gauge_id, wy)  # calculate all values
    append_results(result_lists, gauge_results)  # append all calculated values

df = pd.DataFrame(result_lists)  # create final dataframe

# load attributes from Caravan
df_attributes_hydroatlas = pd.read_csv("D:/Data/Caravan_May25/attributes/camelsaus/attributes_hydroatlas_camelsaus.csv", sep=',')
df_attributes_caravan = pd.read_csv("D:/Data/Caravan_May25/attributes/camelsaus/attributes_caravan_camelsaus.csv", sep=',')
df_attributes_other = pd.read_csv("D:/Data/Caravan_May25/attributes/camelsaus/attributes_other_camelsaus.csv", sep=',')
df = df.merge(df_attributes_hydroatlas, on='gauge_id', how='left')
df = df.merge(df_attributes_caravan, on='gauge_id', how='left')
df = df.merge(df_attributes_other, on='gauge_id', how='left')

# save results
#df.to_csv(results_path + 'camels_AUS_sensitivities_SILO.csv', index=False)
df.to_csv(results_path + 'camels_AUS_sensitivities.csv', index=False)
print("Finished saving data.")

#df = pd.read_csv(results_path + 'camels_AUS_sensitivities_SILO.csv')
df = pd.read_csv(results_path + 'camels_AUS_sensitivities.csv')

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