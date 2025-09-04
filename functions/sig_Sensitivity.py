import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import functions.util_Turc as util_Turc


def sig_Sensitivity(Q, t, P, PET, wateryear="A-SEP", plot_results=False, fit_intercept=False, use_delta=False):
    """
    Calculate sensitivities using multivariate regression with statsmodels OLS.
    Returns sensitivities for P, PET, R2, number of years, VIF, correlation, and p-values for P and PET only.
    """
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("Q, P, PET must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)
    ):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
    df_annual = df_annual.dropna()
    nr_years = len(df_annual)

    if len(df_annual) < 10:
        sens_P = np.NaN
        sens_PET = np.NaN
        R2 = np.NaN
        nr_years = len(df_annual)
        VIF = np.NaN
        corr = np.NaN
        pval_P = np.NaN
        pval_PET = np.NaN
    else:
        if use_delta:
            X = df_annual[["PET", "P"]] - df_annual[["PET", "P"]].mean()
            y = df_annual["Q"] - df_annual["Q"].mean()
        else:
            X = df_annual[["PET", "P"]]
            y = df_annual["Q"]

        if fit_intercept:
            X = add_constant(X)
        else:
            X = X

        model = sm.OLS(y, X).fit()
        sens_P = model.params['P']
        sens_PET = model.params['PET']
        pval_P = model.pvalues['P']
        pval_PET = model.pvalues['PET']
        y_pred = model.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()  # residual sum of squares
        ss_tot = ((y - y.mean()) ** 2).sum()  # total sum of squares relative to mean
        R2 = 1 - ss_res / ss_tot

        #R2 = model.rsquared # uses uncentered R2 which is usually higher

        '''
        from sklearn.linear_model import LinearRegression
        mlg_model = LinearRegression(fit_intercept=fit_intercept)
        mlg_model.fit(X, y)
        sens_P = mlg_model.coef_[1]
        sens_PET = mlg_model.coef_[0]
        R2 = mlg_model.score(X, y)
        '''

        VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        corr = np.corrcoef(X["P"], X["PET"])[0, 1]

        if plot_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
            scatter1 = ax1.scatter(X["P"], y, c=X["PET"], s=50, cmap='viridis', alpha=0.5)
            ax1.scatter(X["P"], model.predict(X), c=X["PET"], marker='d', s=50, cmap='viridis', alpha=0.8)
            ax1.set_xlabel("P [mm]")
            ax1.set_ylabel("Q [mm]")
            ax1.set_title(f"dQ/dP = {sens_P:.2f}")
            fig.colorbar(scatter1, ax=ax1, label="PET [mm]")

            scatter2 = ax2.scatter(X["PET"], y, c=X["P"], s=50, cmap='plasma', alpha=0.5)
            ax2.scatter(X["PET"], model.predict(X), c=X["P"], marker='d', s=50, cmap='plasma', alpha=0.8)
            ax2.set_xlabel("PET [mm]")
            ax2.set_ylabel("Q [mm]")
            ax2.set_title(f"dQ/dPET = {sens_PET:.2f}")
            fig.colorbar(scatter2, ax=ax2, label="P [mm]")

            plt.show()

    return sens_P, sens_PET, R2, nr_years, VIF, corr, pval_P, pval_PET


def sig_SensitivityAveraging(Q, t, P, PET, wateryear="A-SEP", n=5, plot_results=False, fit_intercept=False, use_delta=False):
    """
    Same as sig_Sensitivity but works on n-year block averages before regression.
    Returns the same values including p-values for P and PET.
    """
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("Q, P, PET must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)
    ):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
    df_annual = df_annual.dropna()
    nr_years = len(df_annual)

    # Calculate n-year block averages
    df_block = df_annual.copy().reset_index(drop=True)
    n_blocks = len(df_block) // n
    df_block = df_block.iloc[:n_blocks * n]
    df_block['block'] = np.repeat(np.arange(n_blocks), n)
    df_block_avg = df_block.groupby('block')[['Q', 'P', 'PET']].mean()

    if len(df_block_avg) < 3:
        print("Less than 3 points for chosen block size. Sensitivity not calculated.")
        sens_P = np.NaN
        sens_PET = np.NaN
        R2 = np.NaN
        nr_years = len(df_annual)
        VIF = np.NaN
        corr = np.NaN
        pval_P = np.NaN
        pval_PET = np.NaN
    else:
        if use_delta:
            X = df_block_avg[["PET", "P"]] - df_block_avg[["PET", "P"]].mean()
            y = df_block_avg["Q"] - df_block_avg["Q"].mean()
        else:
            X = df_block_avg[["PET", "P"]]
            y = df_block_avg["Q"]

        if fit_intercept:
            X = add_constant(X)
        else:
            X = X

        model = sm.OLS(y, X).fit()
        sens_P = model.params['P']
        sens_PET = model.params['PET']
        pval_P = model.pvalues['P']
        pval_PET = model.pvalues['PET']
        y_pred = model.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()  # residual sum of squares
        ss_tot = ((y - y.mean()) ** 2).sum()  # total sum of squares relative to mean
        R2 = 1 - ss_res / ss_tot

        VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        corr = np.corrcoef(X["P"], X["PET"])[0, 1]

        if plot_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
            scatter1 = ax1.scatter(X["P"], y, c=X["PET"], s=50, cmap='viridis', alpha=0.8)
            ax1.set_xlabel("P [mm]")
            ax1.set_ylabel("Q [mm]")
            ax1.set_title(f"dQ/dP = {sens_P:.2f}")
            fig.colorbar(scatter1, ax=ax1, label="PET [mm]")
            scatter2 = ax2.scatter(X["PET"], y, c=X["P"], s=50, cmap='plasma', alpha=0.8)
            ax2.set_xlabel("PET [mm]")
            ax2.set_ylabel("Q [mm]")
            ax2.set_title(f"dQ/dPET = {sens_PET:.2f}")
            fig.colorbar(scatter2, ax=ax2, label="P [mm]")
            plt.show()

    return sens_P, sens_PET, R2, nr_years, VIF, corr, pval_P, pval_PET


def sig_SensitivityLog(Q, t, P, PET, wateryear="A-SEP", plot_results=False, fit_intercept=False, use_delta=False):
    """
    Log-log regression to estimate sensitivities, returns p-values for P and PET coefficients.
    """
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("Q, P, PET must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)
    ):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
    df_annual = df_annual.dropna()
    nr_years = len(df_annual)

    if len(df_annual) < 10:
        sens_P = np.NaN
        sens_PET = np.NaN
        R2 = np.NaN
        nr_years = len(df_annual)
        VIF = np.NaN
        corr = np.NaN
        pval_P = np.NaN
        pval_PET = np.NaN
    else:
        if use_delta:
            X = df_annual[["PET", "P"]] - df_annual[["PET", "P"]].mean()
            y = df_annual["Q"] - df_annual["Q"].mean()
        else:
            X = df_annual[["PET", "P"]]
            y = df_annual["Q"]

        try:
            log_X = np.log(X)
            log_y = np.log(y)

            if fit_intercept:
                log_X = add_constant(log_X)
            else:
                log_X = log_X

            model = sm.OLS(log_y, log_X).fit()
            sens_P = model.params['P'] * np.mean(y) / np.mean(X["P"])
            sens_PET = model.params['PET'] * np.mean(y) / np.mean(X["PET"])
            pval_P = model.pvalues['P']
            pval_PET = model.pvalues['PET']
            y_pred = model.predict(log_X)
            ss_res = ((np.exp(log_y) - np.exp(y_pred)) ** 2).sum()  # residual sum of squares
            ss_tot = ((np.exp(log_y) - np.exp(log_y.mean())) ** 2).sum()  # total sum of squares relative to mean
            R2 = 1 - ss_res / ss_tot

            VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

            corr = np.corrcoef(X["P"], X["PET"])[0, 1]

            if plot_results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
                scatter1 = ax1.scatter(np.log(X["P"]), np.log(y), c=np.log(X["PET"]), s=50, cmap='viridis', alpha=0.8)
                ax1.set_xlabel("log P [mm]")
                ax1.set_ylabel("log Q [mm]")
                ax1.set_title(f"dQ/dP = {sens_P:.2f}")
                fig.colorbar(scatter1, ax=ax1, label="log PET [mm]")

                scatter2 = ax2.scatter(np.log(X["PET"]), np.log(y), c=np.log(X["P"]), s=50, cmap='plasma', alpha=0.8)
                ax2.set_xlabel("log PET [mm]")
                ax2.set_ylabel("log Q [mm]")
                ax2.set_title(f"dQ/dPET = {sens_PET:.2f}")
                fig.colorbar(scatter2, ax=ax2, label="P [mm]")

                plt.show()
        except Exception:
            print('Problems with log fitting.')
            sens_P = np.NaN
            sens_PET = np.NaN
            R2 = np.NaN
            nr_years = len(df_annual)
            VIF = np.NaN
            corr = np.NaN
            pval_P = np.NaN
            pval_PET = np.NaN

    return sens_P, sens_PET, R2, nr_years, VIF, corr, pval_P, pval_PET


def sig_SensitivityBudyko(Q, t, P, PET, wateryear="A-SEP", plot_results=False, fit_intercept=False, use_delta=False):
    """
    Budyko-based sensitivity regression with OLS and p-values for P and PET only.
    """
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("Q, P, PET must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)
    ):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
    df_annual = df_annual.dropna()
    df_annual["Q_Budyko"] = util_Turc.calculate_streamflow(df_annual["P"].values, df_annual["PET"].values, 2)
    nr_years = len(df_annual)

    if len(df_annual) < 10:
        sens_P = np.NaN
        sens_PET = np.NaN
        R2 = np.NaN
        nr_years = len(df_annual)
        VIF = np.NaN
        corr = np.NaN
        pval_P = np.NaN
        pval_PET = np.NaN
    else:
        if use_delta:
            X = df_annual[["P", "PET"]] - df_annual[["P", "PET"]].mean()
            y = df_annual["Q_Budyko"] - df_annual["Q_Budyko"].mean()
        else:
            X = df_annual[["P", "PET"]]
            y = df_annual["Q_Budyko"]

        if fit_intercept:
            X = add_constant(X)
        else:
            X = X

        model = sm.OLS(y, X).fit()
        sens_P = model.params['P']
        sens_PET = model.params['PET']
        pval_P = model.pvalues['P']
        pval_PET = model.pvalues['PET']
        y_pred = model.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()  # residual sum of squares
        ss_tot = ((y - y.mean()) ** 2).sum()  # total sum of squares relative to mean
        R2 = 1 - ss_res / ss_tot

        VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        corr = np.corrcoef(X["P"], X["PET"])[0, 1]

        if plot_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
            scatter1 = ax1.scatter(X["P"], y, c=X["PET"], s=50, cmap='viridis', alpha=0.8)
            ax1.set_xlabel("P [mm]")
            ax1.set_ylabel("Q Budyko [mm]")
            ax1.set_title(f"dQ/dP = {sens_P:.2f}")
            fig.colorbar(scatter1, ax=ax1, label="PET [mm]")

            scatter2 = ax2.scatter(X["PET"], y, c=X["P"], s=50, cmap='plasma', alpha=0.8)
            ax2.set_xlabel("PET [mm]")
            ax2.set_ylabel("Q Budyko [mm]")
            ax2.set_title(f"dQ/dPET = {sens_PET:.2f}")
            fig.colorbar(scatter2, ax=ax2, label="P [mm]")

            plt.show()

    return sens_P, sens_PET, R2, nr_years, VIF, corr, pval_P, pval_PET


def sig_SensitivityOverTime(Q, t, P, PET, id=None, window_years=10, wateryear="A-SEP", plot_results=False, fit_intercept=False, use_delta=False):
    """
    Calculate hydrologic metrics using moving water year windows with p-values (only for sensitivities).
    """

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df['wateryear'] = df["t"].dt.to_period(wateryear)
    water_years = df['wateryear'].unique()
    results = []

    if len(water_years) < window_years:
        print("Not enough water years for the specified window size.")
        results.append({
            'start_wateryear': np.nan,
            'end_wateryear': np.nan,
            'mean_P': np.nan,
            'mean_PET': np.nan,
            'mean_Q': np.nan,
            'aridity': np.nan,
            'P_PET_correlation': np.nan,
            'sens_P': np.nan,
            'sens_PET': np.nan,
            'R2': np.nan,
            'pval_sens_P': np.nan,
            'pval_sens_PET': np.nan,
            'nr_years': np.nan,
            'VIF': np.nan
        })
    else:
        for i in range(len(water_years) - window_years + 1):
            start_year = water_years[i]
            end_year = water_years[i + window_years - 1]

            mask = (df['wateryear'] >= start_year) & (df['wateryear'] <= end_year)
            window_df = df[mask]

            mean_P = window_df['P'].mean()
            mean_PET = window_df['PET'].mean()
            mean_Q = window_df['Q'].mean()
            aridity = mean_PET / mean_P if mean_P != 0 else np.nan

            sens_P, sens_PET, R2, nr_years, VIF, corr, pval_P, pval_PET = sig_Sensitivity(
                window_df["Q"].values, window_df["t"].values,
                window_df["P"].values, window_df["PET"].values,
                plot_results=False, use_delta=use_delta, fit_intercept=fit_intercept)

            results.append({
                'start_wateryear': start_year,
                'end_wateryear': end_year,
                'mean_P': mean_P,
                'mean_PET': mean_PET,
                'mean_Q': mean_Q,
                'aridity': aridity,
                'P_PET_correlation': corr,
                'sens_P': sens_P,
                'sens_PET': sens_PET,
                'R2': R2,
                'pval_sens_P': pval_P,
                'pval_sens_PET': pval_PET,
                'nr_years': nr_years,
                'VIF': VIF
            })

    df_results = pd.DataFrame(results)

    if plot_results:
        try:
            fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

            axes[0].plot(df_results['start_wateryear'].dt.year, df_results['mean_P'], label='P', color='tab:blue')
            axes[0].plot(df_results['start_wateryear'].dt.year, df_results['mean_PET'], label='PET', color='tab:orange')
            axes[0].plot(df_results['start_wateryear'].dt.year, df_results['mean_Q'], label='Q', color='tab:purple')

            df_results['Q_Budyko'] = util_Turc.calculate_streamflow(df_results['mean_P'].values, df_results['mean_PET'].values, 2)
            axes[0].plot(df_results['start_wateryear'].dt.year, df_results['Q_Budyko'], label='Q Budyko', color='tab:purple', linestyle='--')
            axes[0].set_ylabel('Flux [mm/year]')
            axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

            axes[1].plot(df_results['start_wateryear'].dt.year, df_results['sens_P'], label='dQ/dP', color='tab:blue')
            axes[1].plot(df_results['start_wateryear'].dt.year, df_results['sens_PET'], label='dQ/dPET', color='tab:orange')

            df_results['sens_P_Budyko'], df_results["sens_PET_Budyko"] = util_Turc.calculate_sensitivities(
                df_results['mean_P'].values, df_results['mean_PET'].values, 2)
            axes[1].plot(df_results['start_wateryear'].dt.year, df_results['sens_P_Budyko'], label='dQ/dP Budyko',
                         color='tab:blue', linestyle='--')
            axes[1].plot(df_results['start_wateryear'].dt.year, df_results['sens_PET_Budyko'], label='dQ/dPET Budyko',
                         color='tab:orange', linestyle='--')
            axes[1].axhline(0, color='darkgrey', linestyle='--')
            axes[1].set_xlabel('Water Year')
            axes[1].set_ylabel('Sensitivity [-]')
            axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

            ax3 = axes[2]
            ax3.plot(df_results['start_wateryear'].dt.year, df_results['aridity'], label='PET/P', color='tab:red')
            ax3.set_ylabel('Aridity [-]', color='tab:red')
            ax3.tick_params(axis='y', labelcolor='tab:red')
            ax3_corr = ax3.twinx()
            ax3_corr.plot(df_results['start_wateryear'].dt.year, df_results['P_PET_correlation'], label='Cor(P,PET)', color='tab:purple')
            ax3_corr.set_ylabel('Correlation [-]', color='tab:purple')
            ax3_corr.tick_params(axis='y', labelcolor='tab:purple')
            ax3.set_xlabel('Water Year')

            fig.tight_layout()
            plt.show()
        except Exception as e:
            print("Plotting didn't work:", e)

    return df_results


def sig_SensitivityWithStorage(Q, t, P, PET, wateryear="A-SEP", plot_results=False, fit_intercept=False, use_delta=False):
    """
    Sensitivity with lagged storage term regression and p-values for sensitivities.
    """
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("Q, P, PET must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)
    ):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
    df_annual["Qlag1"] = df_annual["Q"].shift(1)
    df_annual = df_annual.dropna()
    nr_years = len(df_annual)

    if len(df_annual) < 10:
        sens_P = np.NaN
        sens_PET = np.NaN
        sens_Q = np.NaN
        R2 = np.NaN
        nr_years = len(df_annual)
        pval_P = np.NaN
        pval_PET = np.NaN
        pval_Qlag1 = np.NaN
    else:
        if use_delta:
            X = df_annual[["P", "PET", "Qlag1"]] - df_annual[["P", "PET", "Qlag1"]].mean()
            y = df_annual["Q"] - df_annual["Q"].mean()
        else:
            X = df_annual[["P", "PET", "Qlag1"]]
            y = df_annual["Q"]

        if fit_intercept:
            X = add_constant(X)
        else:
            X = X

        model = sm.OLS(y, X).fit()
        sens_P = model.params['P']
        sens_PET = model.params['PET']
        sens_Q = model.params['Qlag1']
        pval_P = model.pvalues['P']
        pval_PET = model.pvalues['PET']
        pval_Qlag1 = model.pvalues['Qlag1']
        y_pred = model.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()  # residual sum of squares
        ss_tot = ((y - y.mean()) ** 2).sum()  # total sum of squares relative to mean
        R2 = 1 - ss_res / ss_tot

        #VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        if plot_results:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3), tight_layout=True)
            scatter1 = ax1.scatter(X["P"], y, c=X["PET"], s=X["Qlag1"]*50, cmap='viridis', alpha=0.8)
            ax1.set_xlabel("P [mm]")
            ax1.set_ylabel("Q [mm]")
            ax1.set_title(f"dQ/dP = {sens_P:.2f}")
            fig.colorbar(scatter1, ax=ax1, label="PET [mm]")

            scatter2 = ax2.scatter(X["PET"], y, c=X["P"], s=X["Qlag1"]*50, cmap='plasma', alpha=0.8)
            ax2.set_xlabel("PET [mm]")
            ax2.set_ylabel("Q [mm]")
            ax2.set_title(f"dQ/dPET = {sens_PET:.2f}")
            fig.colorbar(scatter2, ax=ax2, label="P [mm]")

            scatter3 = ax3.scatter(X["Qlag1"], y, s=50, alpha=0.8)
            ax3.set_xlabel("Qlag1 [mm]")
            ax3.set_ylabel("Q [mm]")
            ax3.set_title(f"dQ/dQlag1 = {sens_Q:.2f}")

            plt.show()

    return sens_P, sens_PET, sens_Q, R2, nr_years, pval_P, pval_PET, pval_Qlag1
