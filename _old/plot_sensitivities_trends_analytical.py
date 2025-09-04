import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from functions import util_Turc
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression, Ridge
#mpl.use('TkAgg')

# prepare data
data_path = "D:/Data/"

# check if folders exist
results_path = "../results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "../figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

# Generate correlated random series with trends
np.random.seed(123)
years = np.arange(100)  # 100 years

# Set correlation and noise level
corr = -0.5  # Correlation between P and PET
noise_level = 0.01  # Noise level
n = 2.5

# Generate correlated P and PET time series
mean_P, mean_PET = 700, 500
sd_P, sd_PET = 75, 50
cov_P_PET = corr * sd_P * sd_PET  # Covariance between P and PET
sigma = np.array([[sd_P ** 2, cov_P_PET], [cov_P_PET, sd_PET ** 2]])  # Covariance matrix
mu = np.array([mean_P, mean_PET])  # Mean values for P and PET

# Generate correlated samples for P and PET with trends
trend_P = 0.0
trend_PET = 0.5
samples = multivariate_normal(mu, sigma).rvs(len(years))
P = samples[:, 0] * (1 + trend_P * years / max(years))
PET = samples[:, 1] * (1 + trend_PET * years / max(years))  # Add trend to PET
Q = util_Turc.calculate_streamflow(P, PET, n)

# Add noise to the series
Q += np.random.normal(0, noise_level * mean_P, len(years))
P += np.random.normal(0, noise_level * mean_P, len(years))
PET += np.random.normal(0, noise_level * mean_PET, len(years))

# Create DataFrame
df = pd.DataFrame({'Year': years, 'P': P, 'PET': PET, 'Q': Q})

# Calculate sensitivities for 30-year moving blocks
block_size = 30
sens_P_theoretical = []
sens_PET_theoretical = []
sens_P_noncentered = []
sens_PET_noncentered = []
sens_P_centered = []
sens_PET_centered = []

for start in range(len(years) - block_size + 1):
    block = df.iloc[start:start + block_size]

    # Theoretical sensitivities based on mean P and PET in the block
    mean_P_block = block['P'].mean()
    mean_PET_block = block['PET'].mean()
    sens_P, sens_PET = util_Turc.calculate_sensitivities(mean_P_block, mean_PET_block, n)
    sens_P_theoretical.append(sens_P)
    sens_PET_theoretical.append(sens_PET)

    # Method: Non-centered multiple regression
    X_noncentered = block[["P", "PET"]]
    y_noncentered = block["Q"]
    model_noncentered = LinearRegression(fit_intercept=False).fit(X_noncentered, y_noncentered)
    sens_P_nc, sens_PET_nc = model_noncentered.coef_
    sens_P_noncentered.append(sens_P_nc)
    sens_PET_noncentered.append(sens_PET_nc)

    # Method: Centered multiple regression
    X_centered = X_noncentered - X_noncentered.mean()
    y_centered = y_noncentered - y_noncentered.mean()
    model_centered = LinearRegression(fit_intercept=False).fit(X_centered, y_centered)
    sens_P_c, sens_PET_c = model_centered.coef_
    sens_P_centered.append(sens_P_c)
    sens_PET_centered.append(sens_PET_c)

# Create DataFrame for sensitivities
df_sens = pd.DataFrame({
    'Start_Year': years[:len(sens_P_theoretical)],
    'Sens_P_Theoretical': sens_P_theoretical,
    'Sens_PET_Theoretical': sens_PET_theoretical,
    'Sens_P_NonCentered': sens_P_noncentered,
    'Sens_PET_NonCentered': sens_PET_noncentered,
    'Sens_P_Centered': sens_P_centered,
    'Sens_PET_Centered': sens_PET_centered
})

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

# Plot P and PET trends
axs[0].plot(df['Year'], df['P'], label='P', color='tab:blue')
axs[0].plot(df['Year'], df['PET'], label='PET', color='tab:orange')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Values [mm/yr]')
axs[0].set_xlim(0, 100)

# Plot sensitivities (theoretical vs non-centered and centered methods)
axs[1].plot(df_sens['Start_Year'], df_sens['Sens_P_Theoretical'], label='Sens. P (Theoretical)', color='tab:blue',
            linestyle='-')
axs[1].plot(df_sens['Start_Year'], df_sens['Sens_PET_Theoretical'], label='Sens. PET (Theoretical)', color='tab:orange',
            linestyle='-')
axs[1].plot(df_sens['Start_Year'], df_sens['Sens_P_NonCentered'], label='Sens. P (Non-Centered)', color='tab:blue',
            linestyle='--')
axs[1].plot(df_sens['Start_Year'], df_sens['Sens_PET_NonCentered'], label='Sens. PET (Non-Centered)',
            color='tab:orange', linestyle='--')
axs[1].plot(df_sens['Start_Year'], df_sens['Sens_P_Centered'], label='Sens. P (Centered)', color='tab:blue',
            linestyle='-.')
axs[1].plot(df_sens['Start_Year'], df_sens['Sens_PET_Centered'], label='Sens. PET (Centered)', color='tab:orange',
            linestyle='-.')
axs[1].axhline(0, color='darkgrey', linestyle='--')
axs[1].set_xlabel('Start Year of Block')
axs[1].set_ylabel('Sensitivity [-]')
axs[1].set_ylim(-1, 1)

# Move legend outside the plots
handles_0, labels_0 = axs[0].get_legend_handles_labels()
handles_1, labels_1 = axs[1].get_legend_handles_labels()

fig.legend(handles_0 + handles_1, labels_0 + labels_1,
           loc="center right", bbox_to_anchor=(0.85, 0.5), title="Legend")

plt.tight_layout(rect=[0, 0, 0.6, 1])  # Adjust layout to make space for the legend

fig.savefig(figures_path + 'temporal_sensitivities_analytical.png', dpi=600, bbox_inches='tight')

