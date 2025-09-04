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

# todo: change paths etc.

df_names = pd.read_csv(data_path + "CAMELS_US/camels_attributes_v2.0/camels_attributes_v2.0/camels_name.txt",
                       sep=';', skiprows=0, encoding='latin-1')
df_topo = pd.read_csv(data_path + "CAMELS_US/camels_attributes_v2.0/camels_attributes_v2.0/camels_topo.txt",
                      sep=';', skiprows=0, encoding='latin-1')
df_climate = pd.read_csv(data_path + "CAMELS_US/camels_attributes_v2.0/camels_attributes_v2.0/camels_clim.txt",
                         sep=';', skiprows=0, encoding='latin-1')
df_soil = pd.read_csv(data_path + "CAMELS_US/camels_attributes_v2.0/camels_attributes_v2.0/camels_soil.txt",
                      sep=';', skiprows=0, encoding='latin-1')
df_geol = pd.read_csv(data_path + "CAMELS_US/camels_attributes_v2.0/camels_attributes_v2.0/camels_geol.txt",
                      sep=';', skiprows=0, encoding='latin-1')
df_vege = pd.read_csv(data_path + "CAMELS_US/camels_attributes_v2.0/camels_attributes_v2.0/camels_vege.txt",
                      sep=';', skiprows=0, encoding='latin-1')
df_hydro = pd.read_csv(data_path + "CAMELS_US/camels_attributes_v2.0/camels_attributes_v2.0/camels_hydro.txt",
                       sep=';', skiprows=0, encoding='latin-1')

df_attr = pd.merge(df_names, df_topo, on='gauge_id')
df_attr = pd.merge(df_attr, df_climate, on='gauge_id')
df_attr = pd.merge(df_attr, df_soil, on='gauge_id')
df_attr = pd.merge(df_attr, df_geol, on='gauge_id')
df_attr = pd.merge(df_attr, df_vege, on='gauge_id')
df_attr = pd.merge(df_attr, df_hydro, on='gauge_id')

# list to store all attributes
result_lists = initialize_result_lists()

for path, subdirs, files in os.walk(
        data_path + "CAMELS_US/basin_timeseries_v1p2_modelOutput_daymet/model_output_daymet/model_output/flow_timeseries/daymet/"):
    for name in files:
        if '05_model_output' in name:
            df_tmp = pd.read_csv(os.path.join(path, name), sep=r'\s+')
            id = name[0:8]
            # camels_01516500
            print(id)
            gauge_id = f"camels_{id}"
            if gauge_id in result_lists["gauge_id"]:
                print('existing twice')
                continue
            df_tmp.loc[df_tmp["OBS_RUN"] < 0, "OBS_RUN"] = np.nan  # nan is -999 in CAMELS
            df_tmp["date"] = pd.to_datetime(dict(year=df_tmp.YR, month=df_tmp.MNTH, day=df_tmp.DY))

            # rename variables
            df_tmp = df_tmp.rename(columns={"OBS_RUN": "Q","PRCP": "P","PET": "PET","TAIR": "T"})

            # rename variables
            df_tmp = df_tmp.rename(columns={"OBS_RUN": "Q",
                                            "PRCP": "P",
                                            "PET": "PET",
                                            "TAIR": "T"})

            # calculate signatures
            wy = 10  # define water year
            gauge_results = calculate_metrics(df_tmp, id, gauge_id, wy)  # calculate all values
            append_results(result_lists, gauge_results)  # append all calculated values

df = pd.DataFrame(result_lists)  # create final dataframe

# load attributes from Caravan
df_attributes_hydroatlas = pd.read_csv("D:/Data/Caravan_May25/attributes/camels/attributes_hydroatlas_camels.csv", sep=',')
df_attributes_caravan = pd.read_csv("D:/Data/Caravan_May25/attributes/camels/attributes_caravan_camels.csv", sep=',')
df_attributes_other = pd.read_csv("D:/Data/Caravan_May25/attributes/camels/attributes_other_camels.csv", sep=',')
df = df.merge(df_attributes_hydroatlas, on='gauge_id', how='left')
df = df.merge(df_attributes_caravan, on='gauge_id', how='left')
df = df.merge(df_attributes_other, on='gauge_id', how='left')

# save results
df.to_csv(results_path + 'camels_US_sensitivities.csv', index=False)
print("Finished saving data.")

df = pd.read_csv(results_path + 'camels_US_sensitivities.csv')

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