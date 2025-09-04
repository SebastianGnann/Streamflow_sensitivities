import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Assuming X has two predictors
predictor1 = X.columns[0]
predictor2 = X.columns[1]
target = y.name

# Create meshgrid for prediction surface
x1_pred = np.linspace(X[predictor1].min(), X[predictor1].max(), 30)
x2_pred = np.linspace(X[predictor2].min(), X[predictor2].max(), 30)
xx1, xx2 = np.meshgrid(x1_pred, x2_pred)
model_viz = pd.DataFrame({predictor1: xx1.flatten(), predictor2: xx2.flatten()})

# Predict surface values
predicted_surface = mlg_model.predict(model_viz).reshape(xx1.shape)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Determine the overall min and max values for color scaling
vmin = min(predicted_surface.min(), y.min())
vmax = max(predicted_surface.max(), y.max())

# Plot surface
surf = ax.plot_surface(xx1, xx2, predicted_surface, alpha=0.5, cmap='viridis', vmin=vmin, vmax=vmax)

# Plot actual data points, colored by their values
scatter = ax.scatter(X[predictor1], X[predictor2], y, c=y, cmap='viridis', s=50, vmin=vmin, vmax=vmax)

# Add labels and title
ax.set_xlabel(predictor1)
ax.set_ylabel(predictor2)
ax.set_zlabel(target)
ax.set_title(f'Multiple Linear Regression Fit (R² = {R2:.2f})')

# Add a single colorbar for both predicted and observed values
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Values')

plt.tight_layout()
plt.show()


# Pairplot
sns.pairplot(pd.concat([X, y], axis=1), height=3)
plt.suptitle('Pairplot of Variables', y=1.02)
plt.show()


# Custom function to plot partial regression without index
def custom_partregress(endog, exog, exog_others, ax):
    from statsmodels.stats.outliers_influence import OLSInfluence
    from statsmodels.regression.linear_model import OLS

    y = endog
    x1 = exog
    x_other = exog_others

    # Regress y on x_other
    res_yaxis = OLS(y, x_other).fit().resid
    # Regress x1 on x_other
    res_xaxis = OLS(x1, x_other).fit().resid

    ax.scatter(res_xaxis, res_yaxis)
    ax.set_xlabel(exog.name)
    ax.set_ylabel(f'{endog.name} | Others')

    # Add regression line
    coeffs = np.polyfit(res_xaxis, res_yaxis, deg=1)
    ax.plot(res_xaxis, np.poly1d(coeffs)(res_xaxis), color='r')


# Partial Regression Plots
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

# Plot partial regression plots
custom_partregress(y, X[predictor1], X[[predictor2]], axes[0])
custom_partregress(y, X[predictor2], X[[predictor1]], axes[1])

plt.tight_layout()
plt.show()


import pingouin as pg

partial_corr_P_Q = pg.partial_corr(data=df_annual, x='P', y='Q', covar='PET')
partial_corr_P_PET = pg.partial_corr(data=df_annual, x='PET', y='Q', covar='P')
print("Partial correlation between P and Q, controlling for PET:")
print(partial_corr_P_Q.round(3))
print("\nPartial correlation between Q and PET, controlling for P:")
print(partial_corr_P_PET.round(3))

corr_P_Q = df_annual["P"].corr(df_annual["Q"])
print(f"Correlation between P and Q: {corr_P_Q:.3f}")
corr_P_Q = df_annual["PET"].corr(df_annual["Q"])
print(f"Correlation between PET and Q: {corr_P_Q:.3f}")
corr_P_Q = df_annual["PET"].corr(df_annual["P"])
print(f"Correlation between P and PET: {corr_P_Q:.3f}")

