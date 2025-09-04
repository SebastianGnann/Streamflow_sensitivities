import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import functions.util_TurcPike as util_TurcPike
import functions.plotting_fcts as plotting_fcts

# Theoretical analysis of streamflow sensitivities to P and PET based on the Turc-Mezentsev-Pike model
# https://hess.copernicus.org/articles/23/2339/2019/#&gid=1&pid=1
util_TurcPike.plot_sensitivities()

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

# Now create Q with Turc-Mezentsev-Pike model and get theoretical sensitivities
df_art["Q"] = util_TurcPike.calculate_streamflow(df_art["P"].values, df_art["PET"].values, 2)
df_art["Q"] = df_art["Q"] * (1 + np.random.normal(0, 0.05, len(df_art))) # add noise
sens_P, sens_PET = util_TurcPike.calculate_sensitivities(df_art["P"].mean(), df_art["PET"].mean(), 2)
print(f"Theoretical sensitivity of Q to P: {sens_P:.2f}")
print(f"Theoretical sensitivity of Q to PET: {sens_PET:.2f}")

# Pairplot of artifical data
sns.pairplot(df_art, height=3)
plt.suptitle('Pairplot of Variables', y=1.02)
plt.show()

# Now calculate sensitivities with different methods
# Method 1: use fluctuations around the mean (same as using a regression model with y-intercept)
X = df_art[["P", "PET"]] - df_art[["P", "PET"]].mean()
y = df_art["Q"] - df_art["Q"].mean()
mlg_model = LinearRegression(fit_intercept=False)
mlg_model.fit(X, y)
sens_P_1 = mlg_model.coef_[0]
sens_PET_1 = mlg_model.coef_[1]
R2_1 = mlg_model.score(X, y)
print(f"Sensitivity of Q to P using fluctuations around mean: {sens_P_1:.2f}")
print(f"Sensitivity of Q to PET using fluctuations around mean: {sens_PET_1:.2f}")

# Partial regression plots
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
plotting_fcts.partial_regression_plot(y, X["P"], X[["PET"]], axes[0])
plotting_fcts.partial_regression_plot(y, X["PET"], X[["P"]], axes[1])
plt.tight_layout()
plt.show()

# Method 2: use raw data
X = df_art[["P", "PET"]]
y = df_art["Q"]
mlg_model = LinearRegression(fit_intercept=False)
mlg_model.fit(X, y)
sens_P_2 = mlg_model.coef_[0]
sens_PET_2 = mlg_model.coef_[1]
R2_2 = mlg_model.score(X, y)
print(f"Sensitivity of Q to P using raw data: {sens_P_2:.2f}")
print(f"Sensitivity of Q to PET using raw data: {sens_PET_2:.2f}")

# Partial regression plots
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
plotting_fcts.partial_regression_plot(y, X["P"], X[["PET"]], axes[0])
plotting_fcts.partial_regression_plot(y, X["PET"], X[["P"]], axes[1])
plt.tight_layout()
plt.show()

# Finally also print correlations and partial correlations
partial_corr_P_Q = pg.partial_corr(data=df_art, x='P', y='Q', covar='PET')
partial_corr_PET_Q = pg.partial_corr(data=df_art, x='PET', y='Q', covar='P')
corr_P_Q = df_art["P"].corr(df_art["Q"])
corr_PET_Q = df_art["PET"].corr(df_art["Q"])
corr_P_PET = df_art["PET"].corr(df_art["P"])
print(f"Partial correlation between P and Q, controlling for PET: {partial_corr_P_Q["r"].values[0]:.2f}")
print(f"Partial correlation between PET and Q, controlling for P: {partial_corr_PET_Q["r"].values[0]:.2f}")
print(f"Correlation between P and Q: {corr_P_Q:.2f}")
print(f"Correlation between PET and Q: {corr_PET_Q:.2f}")
print(f"Correlation between P and PET: {corr_P_PET:.2f}")


