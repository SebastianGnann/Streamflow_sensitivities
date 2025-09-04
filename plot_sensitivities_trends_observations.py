import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from functions import util_Turc
from scipy.stats import theilslopes
import re
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

# load data
df_CAMELS_US = pd.read_csv(results_path + 'CAMELS_US_sensitivities.csv')
df_CAMELS_US["country"] = 2
# take all catchments

df_CAMELS_GB = pd.read_csv(results_path + 'CAMELS_GB_sensitivities.csv')
df_CAMELS_GB["country"] = 3
# UKBN
df_UKBN = pd.read_csv("results/UKBN_Station_List_vUKBN2.0_1.csv")
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

df = pd.concat([df_CAMELS_DE, df_CAMELS_AUS], ignore_index=True)

# filter catchments
df = df[df["perc_complete"] > 0.95]
df = df[df["len_years"] > 50]
df = df[df["frac_snow_control"] < 0.2]
df = df[df["mean_P"] > df["mean_Q"]]
df = df.reset_index()

n = 2.5 # Turc-Pike parameter

# choose either Germany or Australia
country_nr = 4# 4 for Germany, 5 for Australia
df = df[df["country"] == country_nr]

# transform data
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
df['start_wateryear_first'] = df['start_wateryear'].apply(lambda x: x[0])
df['end_wateryear_last'] = df['end_wateryear'].apply(lambda x: x[-1])
df = df.reset_index()

# create time series of theoretical sensitivities over same blocks (n=20y)
sens_P_theory_list = []
sens_PET_theory_list = []
for i in range(len(df)):
    P_arr = df.loc[i, 'mean_P_over_time']
    PET_arr = df.loc[i, 'mean_PET_over_time']
    sens_P = []
    sens_PET = []
    for P, PET in zip(P_arr, PET_arr):
        sp, spet = util_Turc.calculate_sensitivities(P, PET, n)
        sens_P.append(sp)
        sens_PET.append(spet)
    sens_P_theory_list.append(np.array(sens_P))
    sens_PET_theory_list.append(np.array(sens_PET))
df['sens_P_over_time_theory'] = sens_P_theory_list
df['sens_PET_over_time_theory'] = sens_PET_theory_list

df['mean_P_over_time'] = df['mean_P_over_time']*365
df['mean_PET_over_time'] = df['mean_PET_over_time']*365
df['mean_Q_over_time'] = df['mean_Q_over_time']*365

# list of variables to analyze
variables = ['sens_P_over_time_mr1', 'sens_PET_over_time_mr1', 'sens_P_over_time_mr2', 'sens_PET_over_time_mr2',
             'sens_P_over_time_theory', 'sens_PET_over_time_theory',
             'aridity_over_time', 'mean_P_over_time', 'mean_PET_over_time', 'mean_Q_over_time']

# Create new columns in df for trend storage
for var in variables:
    df[f'{var}_trend'] = np.nan

# Calculate trends for each catchment using df_temp
for idx, row in df.iterrows():
    gauge_id = row['gauge_id']  # Extract gauge_id for the current catchment
    df_time = df[df['gauge_id'] == gauge_id]  # Filter df_temp for rows corresponding to this gauge_id
    if df_time.empty:  # Skip if no data available for this gauge_id
        print("Empty DataFrame for gauge_id:", gauge_id, "Skipping...")
        continue
    x = df_time['start_wateryear'].values[0]  # Assuming start_wateryear is numeric

    # Calculate trends for each variable
    for var in variables:
        y = df_time[var].values[0]  # Extract values for the variable
        valid = ~np.isnan(y)  # Filter out NaN values
        if sum(valid) >= 2:  # Minimum 2 points required for regression
            try:
                medslope, medintercept, lo_slope, up_slope = theilslopes(y[valid], x[valid])
                df.at[idx, f'{var}_trend'] = medslope  # Store Theil-Sen trend (slope) in the main DataFrame
            except Exception as e:
                print(f"Error calculating trend for gauge {gauge_id}, variable {var}: {e}")

def plot_actual_changes(df, variable, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))

    color_map = {
        'sens_P_over_time_mr1': 'tab:blue',
        'sens_PET_over_time_mr1': 'tab:orange',
        'sens_P_over_time_mr2': 'tab:blue',
        'sens_PET_over_time_mr2': 'tab:orange',
        'mean_P_over_time': 'lightsteelblue',
        'mean_PET_over_time': 'sandybrown',
        'aridity_over_time': 'tab:grey',
        'mean_Q_over_time': 'tab:purple'
    }
    main_color = color_map.get(variable, 'tab:gray')

    # Map variable to its theoretical counterpart, if any
    theory_variable_map = {
        'sens_P_over_time_mr1': 'sens_P_over_time_theory',
        'sens_PET_over_time_mr1': 'sens_PET_over_time_theory',
        'sens_P_over_time_mr2': 'sens_P_over_time_theory',
        'sens_PET_over_time_mr2': 'sens_PET_over_time_theory'
    }
    theory_variable = theory_variable_map.get(variable, None)

    yearly_data = {}
    yearly_theory_data = {}

    for idx, row in df.iterrows():
        gauge_id = row['gauge_id']
        df_time = df[df['gauge_id'] == gauge_id]
        if df_time.empty:
            continue

        x = (df_time['start_wateryear'].values[0] + df_time['end_wateryear'].values[0]) / 2
        y = df_time[variable].values[0]
        valid = (x >= 1980) & (x <= 2010)

        # Use the same logic for actual and theoretical values
        if theory_variable:
            y_theory = row[theory_variable]
        else:
            y_theory = None

        for i, year in enumerate(x[valid]):
            value = y[valid][i]
            if year not in yearly_data:
                yearly_data[year] = []
            yearly_data[year].append(value)

            # Handle theoretical values in the same way
            if theory_variable:
                theory_value = y_theory[valid][i]
                if year not in yearly_theory_data:
                    yearly_theory_data[year] = []
                yearly_theory_data[year].append(theory_value)

    unique_years = sorted(yearly_data.keys())
    median_values = []
    lower_percentile_values = []
    upper_percentile_values = []

    # For theory
    median_theory_values = []
    lower_theory_values = []
    upper_theory_values = []

    for year in unique_years:
        values = np.array(yearly_data[year])
        median_values.append(np.nanmedian(values))
        lower_percentile_values.append(np.nanpercentile(values, 25))
        upper_percentile_values.append(np.nanpercentile(values, 75))

        if theory_variable:
            theory_values = np.array(yearly_theory_data.get(year, [np.nan]))
            median_theory_values.append(np.nanmedian(theory_values))
            lower_theory_values.append(np.nanpercentile(theory_values, 25))
            upper_theory_values.append(np.nanpercentile(theory_values, 75))

    ax.fill_between(unique_years, lower_percentile_values, upper_percentile_values,
                    color=main_color, alpha=0.75, label='25th-75th Percentile')
    ax.plot(unique_years, median_values, color=main_color, linewidth=2, label='Median')

    # Plot theoretical line if applicable
    if theory_variable:
        ax.plot(unique_years, median_theory_values, color=main_color, linestyle='--', linewidth=2,
                label='Theoretical')
        #Optionally fill between percentiles for theory
        #ax.fill_between(unique_years, lower_theory_values, upper_theory_values, color=main_color, alpha=0.3)

    variable_to_label = {
        'mean_Q_over_time': r'$Q$ [mm/y]',
        'mean_P_over_time': r'$P$ [mm/y]',
        'mean_PET_over_time': r'$E_p$ [mm/y]',
        'aridity_over_time': r'$E_p$/$P$ [-]',
        'sens_P_over_time_mr1': r'$s_P$ [-]',
        'sens_PET_over_time_mr1': r'$s_{Ep}$ [-]',
        'sens_P_over_time_mr2': r'$s_P$ [-]',
        'sens_PET_over_time_mr2': r'$s_{Ep}$ [-]'
    }

    # ax.set_xlabel('Average year in 20 y window', fontsize=12)
    ax.set_ylabel(variable_to_label.get(variable, variable), fontsize=12)

    return ax

# overwrite sensitivities with elasticities as a check
#df['sens_P_over_time_mr1'] = df['sens_P_over_time_mr1'] / (df['mean_Q_over_time'] / df['mean_P_over_time'])
#df['sens_PET_over_time_mr1'] = df['sens_PET_over_time_mr1'] / (df['mean_Q_over_time'] / df['mean_PET_over_time'])
#df['sens_P_over_time_mr2'] = df['sens_P_over_time_mr2'] / (df['mean_Q_over_time'] / df['mean_P_over_time'])
#df['sens_PET_over_time_mr2'] = df['sens_PET_over_time_mr2'] / (df['mean_Q_over_time'] / df['mean_PET_over_time'])
#df['sens_P_over_time_theory'] = df['sens_P_over_time_theory'] / (df['mean_Q_over_time'] / df['mean_P_over_time'])
#df['sens_PET_over_time_theory'] = df['sens_PET_over_time_theory'] / (df['mean_Q_over_time'] / df['mean_PET_over_time'])

# plot trends
fig, axes = plt.subplots(2, 3, figsize=(11, 3.5))
plot_vars = ['sens_P_over_time_mr1', 'sens_PET_over_time_mr1', 'mean_Q_over_time', 'mean_P_over_time', 'mean_PET_over_time', 'aridity_over_time']
for var, ax in zip(plot_vars, axes.flatten()):
    plot_actual_changes(df, var, ax=ax)
plt.tight_layout()
plt.show()
if country_nr == 4:
    fig.savefig(figures_path + 'temporal_sensitivities_DE.png', dpi=600, bbox_inches='tight')
elif country_nr == 5:
    fig.savefig(figures_path + 'temporal_sensitivities_AUS.png', dpi=600, bbox_inches='tight')

# print trend magnitudes
print_vars = ['sens_P_over_time_mr1', 'sens_PET_over_time_mr1',
              'sens_P_over_time_theory', 'sens_PET_over_time_theory',
              'mean_P_over_time', 'mean_PET_over_time', 'aridity_over_time', 'mean_Q_over_time']

print("Median trends for each variable (per 50 years):")
for var in print_vars:
    median_trend = df[f'{var}_trend'].median()*50
    print(f"{var}: {median_trend:.2f}")

print_vars = ['sens_P_over_time_mr1', 'sens_PET_over_time_mr1',
              'sens_P_over_time_theory', 'sens_PET_over_time_theory']

print("Median normalized relative trends for each variable (per 50 years, normalized to initial value):")
for var in print_vars:
    trends = df[f'{var}_trend'] * 50  # Series of trends per catchment
    initial_values = df[var].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    relative_trends = trends / initial_values.replace(0, np.nan)
    median_relative_trend = relative_trends.median()
    print(f"{var}: {median_relative_trend:.2f}")
