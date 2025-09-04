import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats import linregress
import seaborn as sns
import functions.util_TurcPike as util_TurcPike
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

# Prepare data paths
data_path = "D:/Data/"
results_path = "../results/"
figures_path = "../figures/"

# Ensure folders exist
os.makedirs(results_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Load data
df_temp = pd.read_csv(results_path + 'camels_DE_temporal_sensitivities.csv')
df = pd.read_csv(results_path + 'camels_DE_sensitivities_time.csv')

# Clean up df based on record length and completeness criteria
df = df.loc[df["record_length"] >= 30]
df = df.loc[df["perc_complete"] >= 0.95]
df_ROBIN = pd.read_csv("D:/Python/ROBIN_CAMELS_DE/results/camels_de_ROBIN.csv")
#df = df[df["gauge_id"].isin(df_ROBIN["ID"].values)]

# List of variables to analyze
variables = ['sens_P', 'sens_PET', 'aridity', 'mean_P', 'mean_PET', 'mean_Q']

# Create new columns in df for trend storage
for var in variables:
    df[f'{var}_trend'] = np.nan

# Calculate trends for each catchment using df_temp
for idx, row in df.iterrows():
    gauge_id = row['gauge_id']  # Extract gauge_id for the current catchment

    # Filter df_temp for rows corresponding to this gauge_id
    df_time = df_temp[df_temp['gauge_id'] == gauge_id]

    if df_time.empty:  # Skip if no data available for this gauge_id
        continue

    # Convert water years to numeric values for regression
    x = df_time['start_wateryear'].values.astype(float)  # Assuming start_wateryear is numeric

    # Calculate trends for each variable
    for var in variables:
        y = df_time[var].values  # Extract values for the variable

        valid = ~np.isnan(y)  # Filter out NaN values

        if sum(valid) >= 2:  # Minimum 2 points required for regression
            try:
                slope, intercept, r_value, p_value, std_err = linregress(x[valid], y[valid])
                df.at[idx, f'{var}_trend'] = slope  # Store trend (slope) in the main DataFrame
            except Exception as e:
                print(f"Error calculating trend for gauge {gauge_id}, variable {var}: {e}")


def plot_actual_changes(variable, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    y_axis_ranges = {
        'sens_P': [0, 1.2],
        'sens_PET': [-0.8, 0.4],
        'aridity': [0.4, 1.2],
        'mean_P': [1.5, 3.5],
        'mean_PET': [1.4, 2.2],
        'mean_Q': [0, 2]
    }

    yearly_data = {}
    P_values = {}
    PET_values = {}

    for idx, row in df.iterrows():
        gauge_id = row['gauge_id']

        df_time = df_temp[df_temp['gauge_id'] == gauge_id]

        if df_time.empty:
            continue

        x = (df_time['start_wateryear'].values.astype(float) + df_time['end_wateryear'].values.astype(float)) / 2
        y = df_time[variable].values
        P = df_time['mean_P'].values
        PET = df_time['mean_PET'].values

        valid = ~np.isnan(y)
        # remove anything before 1960
        valid = valid & (x >= 1970)
        valid = valid & (x <= 2010)

        if sum(valid) >= 2:
            ax.plot(x[valid], y[valid], color='gray', alpha=0.2, linewidth=0.5)

            for year, value, p_value, pet_value in zip(x[valid], y[valid], P[valid], PET[valid]):
                if year not in yearly_data:
                    yearly_data[year] = []
                    P_values[year] = []
                    PET_values[year] = []
                yearly_data[year].append(value)
                P_values[year].append(p_value)
                PET_values[year].append(pet_value)

    unique_years = sorted(yearly_data.keys())
    median_values = []
    lower_percentile_values = []
    upper_percentile_values = []
    sens_P_theor_values = []
    sens_PET_theor_values = []

    for year in unique_years:
        values = np.array(yearly_data[year])
        P_median = np.nanmedian(P_values[year])
        PET_median = np.nanmedian(PET_values[year])
        sens_P_theor, sens_PET_theor = util_TurcPike.calculate_sensitivities(P_median, PET_median,n=2)
        sens_P_theor_values.append(sens_P_theor)
        sens_PET_theor_values.append(sens_PET_theor)
        median_values.append(np.nanmedian(values))
        lower_percentile_values.append(np.nanpercentile(values, 25))
        upper_percentile_values.append(np.nanpercentile(values, 75))

    print(f"Median {variable} values: {median_values}")

    ax.plot(unique_years, median_values, color='tab:orange', linewidth=2, label='Median')
    ax.fill_between(unique_years, lower_percentile_values, upper_percentile_values,
                    color='tab:orange', alpha=0.7, label='25th-75th Percentile')

    if variable == 'sens_P':
        ax.plot(unique_years, sens_P_theor_values, color='tab:orange', linestyle='--', linewidth=1.5,
                label='Theoretical Sens P')
    elif variable == 'sens_PET':
        ax.plot(unique_years, sens_PET_theor_values, color='tab:orange', linestyle='--', linewidth=1.5,
                label='Theoretical Sens PET')

    if variable in y_axis_ranges:
        ax.set_ylim(y_axis_ranges[variable])
    ax.set_xlabel('Average year in 20 y window', fontsize=12)
    ax.set_ylabel(variable, fontsize=12)

    return ax

fig, axes = plt.subplots(3, 2, figsize=(10, 7))
plot_vars = ['sens_P', 'sens_PET', 'mean_P', 'mean_PET', 'aridity', 'mean_Q']
for var, ax in zip(plot_vars, axes.flatten()):
    plot_actual_changes(var, ax=ax)
plt.tight_layout()
plt.show()
fig.savefig(figures_path + 'temporal_sensitivities_CAMELS_DE.png', dpi=300, bbox_inches='tight')

# TODO: check all of this ... t_start and t_end probably need to be shifted by 10y..

# Compare changes in aridity and changes in sensitivities by using a scatter plot
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(df['aridity_trend'], df['sens_P_trend'], alpha=0.5, label='dQ/dP', s=10, color='tab:blue')
ax.scatter(df['aridity_trend'], df['sens_PET_trend'], alpha=0.5, label='dQ/dPET', s=10, color='tab:orange')
# plot regression lines
X = df['aridity_trend'].values.reshape(-1, 1)
y_P = df['sens_P_trend'].values
y_PET = df['sens_PET_trend'].values
# fit regression line
model_P = LinearRegression(fit_intercept=False)
model_P.fit(X, y_P)
slope_P = model_P.coef_[0]
model_PET = LinearRegression(fit_intercept=False)
model_PET.fit(X, y_PET)
slope_PET = model_PET.coef_[0]
x_vals = np.linspace(df['aridity_trend'].min(), df['aridity_trend'].max(), 100).reshape(-1, 1)
ax.plot(x_vals, slope_P * x_vals, color='tab:blue', linestyle='-', label='dQ/dP')
ax.plot(x_vals, slope_PET * x_vals, color='tab:orange', linestyle='-', label='dQ/dPET')
# Calculate theoretical relationship for aridity trend range
t_start = 1970#df_temp['start_wateryear'].min()
t_end = 2010#df_temp['end_wateryear'].max()
P_median = df_temp['mean_P'].median()
PET_median = df_temp['mean_PET'].median()
time_span = t_end - t_start  # Use your actual time span from previous calculation
# Calculate theoretical sensitivity trends using chain rule derivatives
sens_P_deriv = []
sens_PET_deriv = []
for ai_trend in x_vals:
    dAI_dt = ai_trend
    AI = P_median / PET_median  # Current median aridity
    # Calculate partial derivatives using finite differences
    delta = 0.001
    sens_P1, sens_PET1 = util_TurcPike.calculate_sensitivities(P_median * (1 - delta), PET_median, n=2)
    sens_P2, sens_PET2 = util_TurcPike.calculate_sensitivities(P_median * (1 + delta), PET_median, n=2)
    dsensP_dAI = -(sens_P2 - sens_P1) / (2 * delta * P_median / PET_median)
    dsensPET_dAI = -(sens_PET2 - sens_PET1) / (2 * delta * P_median / PET_median)
    sens_P_deriv.append(dsensP_dAI * dAI_dt)
    sens_PET_deriv.append(dsensPET_dAI * dAI_dt)
ax.plot(x_vals, sens_P_deriv, color='tab:blue', linestyle='--', label='Theoretical dQ/dP')
ax.plot(x_vals, sens_PET_deriv, color='tab:orange', linestyle='--', label='Theoretical dQ/dPET')
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.axvline(0, color='grey', linestyle='--', linewidth=1)
ax.set_xlabel('Aridity Trend')
ax.set_ylabel('Sensitivity Trend')
#ax.legend()
ax.set_xlim(-0.002,0.014)
ax.set_ylim(-0.02,0.02)
plt.tight_layout()
plt.show()
fig.savefig(figures_path + 'aridity_sensitivity_trends_CAMELS_DE.png', dpi=300, bbox_inches='tight')

# Categorize BFI5 into three groups
df['BFI_category'] = pd.cut(df['BFI5'], bins=[0, 0.5, 0.75, 1.0], labels=['0-0.5', '0.5-0.75', '0.75-1.0'])
category_colors = {'0-0.5': 'tab:pink', '0.5-0.75': 'tab:cyan', '0.75-1.0': 'tab:blue'}
trend_styles = {'sens_P_trend': '-', 'sens_PET_trend': '--'}
fig, ax = plt.subplots(figsize=(5, 4))
x_vals = np.linspace(df['aridity_trend'].min(), df['aridity_trend'].max(), 100).reshape(-1, 1)
for category, color in category_colors.items():
    subset = df[df['BFI_category'] == category]
    ax.scatter(subset['aridity_trend'], subset['sens_P_trend'], alpha=0.6, s=20, label=f'BFI {category} (dQ/dP)', color=color)
    X = subset['aridity_trend'].values.reshape(-1, 1)
    y = subset['sens_P_trend'].values
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    y_vals = model.predict(x_vals)
    ax.plot(x_vals, y_vals, color=color, linestyle=trend_styles['sens_P_trend'], label=f'Regression (BFI {category}, dQ/dP)')
    ax.scatter(subset['aridity_trend'], subset['sens_PET_trend'], alpha=0.6, s=20, label=f'BFI {category} (dQ/dPET)', color=color, marker='x')
    y_pet = subset['sens_PET_trend'].values
    model_pet = LinearRegression(fit_intercept=False)
    model_pet.fit(X, y_pet)
    y_pet_vals = model_pet.predict(x_vals)
    ax.plot(x_vals, y_pet_vals, color=color, linestyle=trend_styles['sens_PET_trend'], label=f'Regression (BFI {category}, dQ/dPET)')
ax.plot(x_vals, sens_P_deriv, color='tab:grey', linestyle='-', label='Theoretical dQ/dP')
ax.plot(x_vals, sens_PET_deriv, color='tab:grey', linestyle='--', label='Theoretical dQ/dPET')
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.axvline(0, color='grey', linestyle='--', linewidth=1)
ax.set_xlabel('Aridity Trend')
ax.set_ylabel('Sensitivity Trend')
#ax.legend()
ax.set_xlim(-0.002,0.014)
ax.set_ylim(-0.02,0.02)
ax.set_title('Aridity vs Sensitivity Trends by BFI Category')
plt.tight_layout()
plt.show()
fig.savefig(figures_path + 'aridity_sensitivity_trends_by_BFI_with_PET.png', dpi=300, bbox_inches='tight')


# Categorize BFI5 into three groups
df['BFI_category'] = pd.cut(df['BFI5'], bins=[0, 0.66, 1.0], labels=['0-0.66', '0.66-1.0'])
category_colors = {'0-0.66': 'tab:cyan', '0.66-1.0': 'tab:blue'}
trend_styles = {'sens_P_trend': '-', 'sens_PET_trend': '--'}
fig, ax = plt.subplots(figsize=(5, 4))
x_vals = np.linspace(df['aridity_trend'].min(), df['aridity_trend'].max(), 100).reshape(-1, 1)
for category, color in category_colors.items():
    subset = df[df['BFI_category'] == category]
    ax.scatter(subset['aridity_trend'], subset['sens_P_trend'], alpha=0.6, s=20, label=f'BFI {category} (dQ/dP)', color=color)
    X = subset['aridity_trend'].values.reshape(-1, 1)
    y = subset['sens_P_trend'].values
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    y_vals = model.predict(x_vals)
    ax.plot(x_vals, y_vals, color=color, linestyle=trend_styles['sens_P_trend'], label=f'Regression (BFI {category}, dQ/dP)')
    ax.scatter(subset['aridity_trend'], subset['sens_PET_trend'], alpha=0.6, s=20, label=f'BFI {category} (dQ/dPET)', color=color, marker='x')
    y_pet = subset['sens_PET_trend'].values
    model_pet = LinearRegression(fit_intercept=False)
    model_pet.fit(X, y_pet)
    y_pet_vals = model_pet.predict(x_vals)
    ax.plot(x_vals, y_pet_vals, color=color, linestyle=trend_styles['sens_PET_trend'], label=f'Regression (BFI {category}, dQ/dPET)')
ax.plot(x_vals, sens_P_deriv, color='tab:grey', linestyle='-', label='Theoretical dQ/dP')
ax.plot(x_vals, sens_PET_deriv, color='tab:grey', linestyle='--', label='Theoretical dQ/dPET')
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.axvline(0, color='grey', linestyle='--', linewidth=1)
ax.set_xlabel('Aridity Trend')
ax.set_ylabel('Sensitivity Trend')
#ax.legend()
ax.set_xlim(-0.002,0.014)
ax.set_ylim(-0.02,0.02)
ax.set_title('Aridity vs Sensitivity Trends by BFI Category')
plt.tight_layout()
plt.show()
fig.savefig(figures_path + 'aridity_sensitivity_trends_by_BFI_with_PET.png', dpi=300, bbox_inches='tight')

corr = df['sens_P_trend'].corr(df['BFI5'])
print(f'Correlation between sens_P_trend and BFI: {round(corr,2)}')
corr = df['sens_PET_trend'].corr(df['BFI5'])
print(f'Correlation between sens_PET_trend and BFI: {round(corr,2)}')
