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
mpl.use('TkAgg')

np.random.seed(111)

mu = np.array([750, 500])
sd_P, sd_PET = 75, 50
corr = -0.3
cov_P_PET = corr * sd_P * sd_PET
sigma = np.array([[sd_P**2, cov_P_PET], [cov_P_PET, sd_PET**2]])

p_means = np.linspace(200, 2000, 20)
pet_means = np.linspace(200, 2000, 20)
covariances = [-0.5, 0.0, 0.5]
results = []

for p_mean in p_means:
    for pet_mean in pet_means:
        for cov in covariances:
            mu = np.array([p_mean, pet_mean])
            cov_P_PET = cov * sd_P * sd_PET
            sigma = np.array([[sd_P**2, cov_P_PET], [cov_P_PET, sd_PET**2]])
            sample = stats.multivariate_normal(mu, sigma).rvs(50)
            df_art = pd.DataFrame(sample, columns=['P', 'PET'])

            df_art["Q"] = util_TurcPike.calculate_streamflow(df_art["P"].values, df_art["PET"].values, 2)
            df_art["Q"] = df_art["Q"] * (1 + np.random.normal(0, 0.05, len(df_art)))
            sens_P, sens_PET = util_TurcPike.calculate_sensitivities(p_mean, pet_mean, 2)

            # Calculate elasticities
            q_mean = df_art["Q"].mean()
            elas_P = sens_P * (p_mean / q_mean)
            elas_PET = sens_PET * (pet_mean / q_mean)

            aridity_index = pet_mean / p_mean

            X = df_art[["P", "PET"]] - df_art[["P", "PET"]].mean()
            y = df_art["Q"] - df_art["Q"].mean()
            mlg_model = LinearRegression(fit_intercept=False).fit(X, y)

            # Calculate elasticities for Method 1
            elas_P_m1 = mlg_model.coef_[0] * (p_mean / q_mean)
            elas_PET_m1 = mlg_model.coef_[1] * (pet_mean / q_mean)

            results.append({
                'aridity_index': aridity_index,
                'covariance': cov,
                'method': 'Method 1',
                'elas_P': elas_P_m1,
                'elas_PET': elas_PET_m1
            })

            X = df_art[["P", "PET"]]
            y = df_art["Q"]
            mlg_model = LinearRegression(fit_intercept=False).fit(X, y)

            # Calculate elasticities for Method 2
            elas_P_m2 = mlg_model.coef_[0] * (p_mean / q_mean)
            elas_PET_m2 = mlg_model.coef_[1] * (pet_mean / q_mean)

            results.append({
                'aridity_index': aridity_index,
                'covariance': cov,
                'method': 'Method 2',
                'elas_P': elas_P_m2,
                'elas_PET': elas_PET_m2
            })

            results.append({
                'aridity_index': aridity_index,
                'covariance': cov,
                'method': 'Theoretical',
                'elas_P': elas_P,
                'elas_PET': elas_PET
            })

results_df = pd.DataFrame(results)

results_melted = results_df.melt(
    id_vars=['aridity_index', 'covariance', 'method'],
    value_vars=['elas_P', 'elas_PET'],
    var_name='variable', value_name='elasticity'
)

fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

for i, cov in enumerate(covariances):
    ax = axes[i]
    subset = results_melted[results_melted['covariance'] == cov]

    sns.lineplot(
        data=subset,
        x='aridity_index', y='elasticity', hue='method', style='variable',
        markers=False, dashes=False, ax=ax, legend=False,
        palette=['tab:blue', 'tab:orange', 'black']
    )

    ax.set_title(f'Covariance: {cov}')
    ax.set_xlabel('Aridity Index (PET / P)')
    ax.set_ylabel('Elasticity')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_ylim(-3, 4)  # Set y-axis limits from -3 to +4

plt.tight_layout()
plt.show()