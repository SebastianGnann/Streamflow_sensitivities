import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import functions.util_TurcPike as util_TurcPike
import functions.plotting_fcts as plotting_fcts
import matplotlib as mpl
mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer


# Set seed for reproducibility
np.random.seed(111)

# Define artificial P and PET data and covariance matrix
mu = np.array([750, 500])  # Mean of P and PET
sd_P = 75  # Standard deviation for P
sd_PET = 50  # Standard deviation for PET
corr = -0.3  # Desired correlation
cov_P_PET = corr * sd_P * sd_PET
sigma = np.array([[sd_P**2, cov_P_PET], [cov_P_PET, sd_PET**2]])
sample = stats.multivariate_normal(mu, sigma).rvs(50) # Generate random sample
df_art = pd.DataFrame(sample, columns=['P', 'PET']) # Create dataframe
# Add these parameters at the top
p_means = np.linspace(200, 2000, 20)  # P values
pet_means = np.linspace(200, 2000, 20)  # PET values
covariances = [-0.5, 0.0, 0.5]  # covariance values
results = []

# Main analysis loop with aridity index
for p_mean in p_means:
    for pet_mean in pet_means:
        for cov in covariances:
            # Generate artificial data with current parameters
            mu = np.array([p_mean, pet_mean])
            cov_P_PET = cov * sd_P * sd_PET
            sigma = np.array([[sd_P ** 2, cov_P_PET], [cov_P_PET, sd_PET ** 2]])
            sample = stats.multivariate_normal(mu, sigma).rvs(50)
            df_art = pd.DataFrame(sample, columns=['P', 'PET'])

            # Calculate theoretical sensitivities
            df_art["Q"] = util_TurcPike.calculate_streamflow(df_art["P"].values, df_art["PET"].values, 2)
            df_art["Q"] = df_art["Q"] * (1 + np.random.normal(0, 0.05, len(df_art)))  # add noise
            sens_P, sens_PET = util_TurcPike.calculate_sensitivities(p_mean, pet_mean, 2)

            # Calculate aridity index (PET/P)
            aridity_index = pet_mean / p_mean

            # Calculate Method 1
            X = df_art[["P", "PET"]] - df_art[["P", "PET"]].mean()
            y = df_art["Q"] - df_art["Q"].mean()
            mlg_model = LinearRegression(fit_intercept=False).fit(X, y)

            # Store Method 1 results
            results.append({
                'aridity_index': aridity_index,
                'covariance': cov,
                'method': 'Method 1',
                'sens_P': mlg_model.coef_[0],
                'sens_PET': mlg_model.coef_[1]
            })

            # Calculate Method 2
            X = df_art[["P", "PET"]]
            y = df_art["Q"]
            mlg_model = LinearRegression(fit_intercept=False).fit(X, y)

            # Store Method 2 results
            results.append({
                'aridity_index': aridity_index,
                'covariance': cov,
                'method': 'Method 2',
                'sens_P': mlg_model.coef_[0],
                'sens_PET': mlg_model.coef_[1]
            })

            # Store theoretical results
            results.append({
                'aridity_index': aridity_index,
                'covariance': cov,
                'method': 'Theoretical',
                'sens_P': sens_P,
                'sens_PET': sens_PET
            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Melt DataFrame for plotting
results_melted = results_df.melt(
    id_vars=['aridity_index', 'covariance', 'method'],
    value_vars=['sens_P', 'sens_PET'],
    var_name='variable', value_name='sensitivity'
)

# Plot sensitivities as a function of aridity index for each covariance value
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

for i, cov in enumerate(covariances):
    ax = axes[i]
    subset = results_melted[results_melted['covariance'] == cov]

    sns.lineplot(
        data=subset,
        x='aridity_index', y='sensitivity', hue='method', style='variable',
        markers=False, dashes=False, ax=ax, legend=False,
        palette=['tab:blue', 'tab:orange', 'black']
    )

    ax.set_title(f'Covariance: {cov}')
    ax.set_xlabel('Aridity Index (PET / P)')
    ax.set_ylabel('Sensitivity')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xscale('log')
    #ax.legend(title='Method & Variable')

plt.tight_layout()
plt.show()
