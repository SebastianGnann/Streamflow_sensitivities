import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def stepwise_regression_plot(Q, P, PET, t, wateryear="A-SEP", plot_results=True, fit_intercept=False):

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
    df_annual = df_annual.dropna()

    X_noncentered, y_noncentered = df_annual[["P", "PET"]], df_annual["Q"]
    sens_P_method1, sens_PET_method1 = map(round, LinearRegression(fit_intercept=False).fit(X_noncentered, y_noncentered).coef_, [2, 2])
    print(f"Method 1: Sensitivity to P: {sens_P_method1}, Sensitivity to PET: {sens_PET_method1}")
    X_centered, y_centered = df_annual[["P", "PET"]] - df_annual[["P", "PET"]].mean(), df_annual["Q"] - df_annual["Q"].mean()
    sens_P_method2, sens_PET_method2 = map(round, LinearRegression(fit_intercept=False).fit(X_centered, y_centered).coef_, [2, 2])
    print(f"Method 2: Sensitivity to P: {sens_P_method2}, Sensitivity to PET: {sens_PET_method2}")

    # variance inflation factor for non-centered data and centered data
    vif_data = pd.DataFrame()
    X_noncentered = X_noncentered.assign(const=1)
    vif_data["Variable"] = X_noncentered.columns
    vif_data["VIF"] = [variance_inflation_factor(X_noncentered.values, i) for i in range(X_noncentered.shape[1])]
    print("Variance Inflation Factors:")
    print(vif_data)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_centered.columns
    vif_data["VIF"] = [variance_inflation_factor(X_centered.values, i) for i in range(X_centered.shape[1])]
    print("Variance Inflation Factors:")
    print(vif_data)

    # Fit linear regression models for Q vs P and Q vs PET
    fit_in = fit_intercept
    model_P_Q = LinearRegression(fit_intercept=fit_in).fit(df_annual[['P']], df_annual['Q'])
    model_PET_Q = LinearRegression(fit_intercept=fit_in).fit(df_annual[['PET']], df_annual['Q'])
    model_P_PET = LinearRegression(fit_intercept=fit_in).fit(df_annual[['P']], df_annual['PET'])
    model_PET_P = LinearRegression(fit_intercept=fit_in).fit(df_annual[['PET']], df_annual['P'])

    # Calculate residuals
    residual_Q_after_P = df_annual['Q'] - model_P_Q.predict(df_annual[['P']])
    residual_Q_after_PET = df_annual['Q'] - model_PET_Q.predict(df_annual[['PET']])
    residual_PET_after_P = df_annual['PET'] - model_P_PET.predict(df_annual[['P']])
    residual_P_after_PET = df_annual['P'] - model_PET_P.predict(df_annual[['PET']])

    # Partial regression (remove PET from both Q and P etc.)
    partial_regression_P_Q = LinearRegression(fit_intercept=True).fit(residual_P_after_PET.to_numpy().reshape(-1, 1), residual_Q_after_PET)
    sens_partial_P = partial_regression_P_Q.coef_[0]
    partial_regression_PET_Q = LinearRegression(fit_intercept=True).fit(residual_PET_after_P.to_numpy().reshape(-1, 1), residual_Q_after_P)
    sens_partial_PET = partial_regression_PET_Q.coef_[0]
    print(f"Partial Sensitivity to P: {round(sens_partial_P,2)}, Partial Sensitivity to PET: {round(sens_partial_PET,2)}")

    # Semi-partial regression (remove P only from PET etc.)
    semi_partial_regression_P_Q = LinearRegression(fit_intercept=True).fit(residual_P_after_PET.to_numpy().reshape(-1, 1), df_annual['Q'] )
    sens_semi_partial_P = semi_partial_regression_P_Q.coef_[0]
    semi_partial_regression_PET_Q = LinearRegression(fit_intercept=True).fit(residual_PET_after_P.to_numpy().reshape(-1, 1), df_annual['Q'])
    sens_semi_partial_PET = semi_partial_regression_PET_Q.coef_[0]
    print(f"Semi-partial Sensitivity to P: {round(sens_semi_partial_P,2)}, Semi-partial Sensitivity to PET: {round(sens_semi_partial_PET,2)}")

    if plot_results:
        # P vs PET
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].scatter(df_annual[['P']], df_annual[['PET']], color='grey', alpha=0.6)
        axs[0].plot(df_annual[['P']], model_P_PET.predict(df_annual[['P']]), color='grey', label=f'Regression Line (dQ/dP)')
        axs[0].plot(model_PET_P.predict(df_annual[['PET']]), df_annual[['PET']], color='grey', linestyle=':', label=f'Regression Line (dQ/dP)')
        axs[0].set_xlabel('P [mm/yr]')
        axs[0].set_ylabel('PET [mm/yr]')

        axs[1].scatter(df_annual[['P']], df_annual[['Q']], color='tab:blue', alpha=0.6)
        axs[1].scatter(df_annual[['PET']], df_annual[['Q']], color='tab:orange', alpha=0.6)
        axs[1].set_xlabel('P/PET [mm/yr]')
        axs[1].set_ylabel('Q [mm/yr]')

        axs[2].scatter(residual_P_after_PET, residual_Q_after_PET, color='tab:blue', alpha=0.6)
        axs[2].plot(residual_P_after_PET, partial_regression_P_Q.predict(residual_P_after_PET.to_numpy().reshape(-1, 1)), color='tab:blue', label=f'Regression Line (dQ/dP)')
        axs[2].set_xlabel('P/PET residuals [mm/yr]')
        axs[2].scatter(residual_PET_after_P, residual_Q_after_P, color='tab:orange', alpha=0.6)
        axs[2].plot(residual_PET_after_P, partial_regression_PET_Q.predict(residual_PET_after_P.to_numpy().reshape(-1, 1)), color='tab:orange', label=f'Regression Line (dQ/dPET)')
        axs[2].set_ylabel('Q residuals [mm/yr]')
        plt.tight_layout()
        plt.show()

# Method 2 equals partial and semi-partial approach if intercept is used in first regression (independent of second)
# Method 1 equals partial and semi-partial approach if intercept is not used at all
# Method 1 equals partial but not semi-partial approach if intercept is only used in second regression (this leads to different results for the semi-partial regression)



    '''
    # Fit regression lines for residuals (this is not really used and may not make much sense?)
    model_residual_P = LinearRegression(fit_intercept=True).fit(df_annual[['P']], residual_Q_after_PET)
    sens_residual_P = model_residual_P.coef_[0]
    model_residual_PET = LinearRegression(fit_intercept=True).fit(df_annual[['PET']], residual_Q_after_P)
    sens_residual_PET = model_residual_PET.coef_[0]
    print(f"Residual Sensitivity to P: {round(sens_residual_P,2)}, Residual Sensitivity to PET: {round(sens_residual_PET,2)}")
    '''
