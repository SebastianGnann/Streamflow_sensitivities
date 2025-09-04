import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import functions.util_TurcPike as util_TurcPike
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def sig_Sensitivity(Q, t, P, PET, wateryear="A-SEP", plot_results=False, fit_intercept=False, use_delta=False):
    """
    ...

    Parameters:
    Q (array-like): Streamflow [mm/timestep]
    t (array-like): Time [datetime or numeric]
    P (array-like): Precipitation [mm/timestep]
    PET (array-like): Potential evapotranspiration [mm/timestep]
    threshold (float, optional): Temperature threshold to distinguish between rain and snow [°C]. Default is 0.

    Returns:

    """
    # Input validation
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("... must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
    df_annual = df_annual.dropna()

    if len(df_annual) < 10:
        print("Time series shorter than 10y")
        sens_P = np.NaN
        sens_PET = np.NaN
        R2 = np.NaN
        nr_years = len(df_annual)
        VIF = np.NaN
        corr = np.NaN

    else:
        if use_delta:
            X = df_annual[["PET", "P"]] - df_annual[["PET", "P"]].mean()
            y = df_annual["Q"] - df_annual["Q"].mean()
        else:
            X = df_annual[["PET","P"]]
            y = df_annual["Q"]

        mlg_model = LinearRegression(fit_intercept=fit_intercept)
        mlg_model.fit(X, y)

        sens_P = mlg_model.coef_[1]
        sens_PET = mlg_model.coef_[0]
        R2 = mlg_model.score(X, y)
        nr_years = len(df_annual)

        # calculate VIF for predictors
        if fit_intercept:
            X_c = add_constant(X)
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_c.columns
            VIF = [variance_inflation_factor(X_c.values, i) for i in range(X_c.shape[1])]
        else:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        corr = np.corrcoef(X["P"], X["PET"])[0, 1]

        # Optional plotting
        if plot_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
            scatter1 = ax1.scatter(X["P"], y, c=X["PET"], s=50, cmap='viridis', alpha=0.8)
            ax1.set_xlabel("P [mm]")
            ax1.set_ylabel("Q [mm]")
            ax1.set_title("dQ/dP = {:.2f}".format(sens_P))
            cbar1 = fig.colorbar(scatter1, ax=ax1)
            cbar1.set_label("PET [mm]")
            scatter2 = ax2.scatter(X["PET"], y, c=X["P"], s=50, cmap='plasma', alpha=0.8)
            ax2.set_xlabel("PET [mm]")
            ax2.set_ylabel("Q [mm]")
            ax2.set_title("dQ/dPET = {:.2f}".format(sens_PET))
            cbar2 = fig.colorbar(scatter2, ax=ax2)
            cbar2.set_label("P [mm]")
            plt.show()

    return sens_P, sens_PET, R2, nr_years, VIF, corr


def sig_SensitivityAveraging(Q, t, P, PET, wateryear="A-SEP", n=5, plot_results=False, fit_intercept=False, use_delta=False):
    """
    ...

    Parameters:
    Q (array-like): Streamflow [mm/timestep]
    t (array-like): Time [datetime or numeric]
    P (array-like): Precipitation [mm/timestep]
    PET (array-like): Potential evapotranspiration [mm/timestep]
    threshold (float, optional): Temperature threshold to distinguish between rain and snow [°C]. Default is 0.

    Returns:

    """
    # Input validation
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("... must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
    df_annual = df_annual.dropna()

    # Calculate n-year block averages and exclude last period
    df_block = df_annual.copy().reset_index(drop=True)
    n_blocks = len(df_block) // n
    df_block = df_block.iloc[:n_blocks * n]
    df_block['block'] = np.repeat(np.arange(n_blocks), n)
    df_block_avg = df_block.groupby('block')[['Q', 'P', 'PET']].mean()

    if len(df_block_avg) < 3:  # Changed to check block-averaged data
        print("Less than 3 points for chosen block size. Sensitivity not calculated.")
        sens_P = np.NaN
        sens_PET = np.NaN
        R2 = np.NaN
        nr_years = len(df_annual)
        VIF = np.NaN
        corr = np.NaN
    else:
        if use_delta:
            X = df_block_avg[["PET", "P"]] - df_block_avg[["PET", "P"]].mean()
            y = df_block_avg["Q"] - df_block_avg["Q"].mean()
        else:
            X = df_block_avg[["PET","P"]]  # Use block-averaged data
            y = df_block_avg["Q"]

        mlg_model = LinearRegression(fit_intercept=fit_intercept)
        mlg_model.fit(X, y)

        sens_P = mlg_model.coef_[1]
        sens_PET = mlg_model.coef_[0]
        R2 = mlg_model.score(X, y)
        nr_years = len(df_annual)

        # calculate VIF for predictors
        if fit_intercept:
            X_c = add_constant(X)
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_c.columns
            VIF = [variance_inflation_factor(X_c.values, i) for i in range(X_c.shape[1])]
        else:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        corr = np.corrcoef(X["P"], X["PET"])[0, 1]

        # Optional plotting
        if plot_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
            scatter1 = ax1.scatter(X["P"], y, c=X["PET"], s=50, cmap='viridis', alpha=0.8)
            ax1.set_xlabel("P [mm]")
            ax1.set_ylabel("Q [mm]")
            ax1.set_title("dQ/dP = {:.2f}".format(sens_P))
            cbar1 = fig.colorbar(scatter1, ax=ax1)
            cbar1.set_label("PET [mm]")
            scatter2 = ax2.scatter(X["PET"], y, c=X["P"], s=50, cmap='plasma', alpha=0.8)
            ax2.set_xlabel("PET [mm]")
            ax2.set_ylabel("Q [mm]")
            ax2.set_title("dQ/dPET = {:.2f}".format(sens_PET))
            cbar2 = fig.colorbar(scatter2, ax=ax2)
            cbar2.set_label("P [mm]")
            plt.show()

    return sens_P, sens_PET, R2, nr_years, VIF, corr


def sig_SensitivityLog(Q, t, P, PET, wateryear="A-SEP", plot_results=False, fit_intercept=False, use_delta=False):
    """
    ...

    Parameters:
    Q (array-like): Streamflow [mm/timestep]
    t (array-like): Time [datetime or numeric]
    P (array-like): Precipitation [mm/timestep]
    PET (array-like): Potential evapotranspiration [mm/timestep]
    threshold (float, optional): Temperature threshold to distinguish between rain and snow [°C]. Default is 0.

    Returns:

    """
    # Input validation
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("... must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
    df_annual = df_annual.dropna()

    if len(df_annual) < 10:
        print("Time series shorter than 10y")
        sens_P = np.NaN
        sens_PET = np.NaN
        R2 = np.NaN
        nr_years = len(df_annual)
        VIF = np.NaN
        corr = np.NaN

    else:
        if use_delta:
            X = df_annual[["PET", "P"]] - df_annual[["PET", "P"]].mean()
            y = df_annual["Q"] - df_annual["Q"].mean()
        else:
            X = df_annual[["PET","P"]]
            y = df_annual["Q"]

        try:
            mlg_model = LinearRegression(fit_intercept=fit_intercept).fit(np.log(X), np.log(y))

            sens_P = mlg_model.coef_[1]*np.mean(y)/np.mean(X["P"])
            sens_PET = mlg_model.coef_[0]*np.mean(y)/np.mean(X["PET"])
            R2 = mlg_model.score(np.log(X), np.log(y))
            nr_years = len(df_annual)

            # calculate VIF for predictors
            if fit_intercept:
                X_c = add_constant(X)
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X_c.columns
                VIF = [variance_inflation_factor(X_c.values, i) for i in range(X_c.shape[1])]
            else:
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X.columns
                VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

            corr = np.corrcoef(X["P"], X["PET"])[0, 1]

            # Optional plotting
            if plot_results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
                scatter1 = ax1.scatter(np.log(X["P"]), np.log(y), c=np.log(X["PET"]), s=50, cmap='viridis', alpha=0.8)
                ax1.set_xlabel("log P [mm]")
                ax1.set_ylabel("log Q [mm]")
                ax1.set_title("dQ/dP = {:.2f}".format(sens_P))
                cbar1 = fig.colorbar(scatter1, ax=ax1)
                cbar1.set_label("log PET [mm]")
                scatter2 = ax2.scatter(np.log(X["PET"]), np.log(y), c=np.log(X["P"]), s=50, cmap='plasma', alpha=0.8)
                ax2.set_xlabel("log PET [mm]")
                ax2.set_ylabel("log Q [mm]")
                ax2.set_title("dQ/dPET = {:.2f}".format(sens_PET))
                cbar2 = fig.colorbar(scatter2, ax=ax2)
                cbar2.set_label("P [mm]")
                plt.show()
        except:
            print('Problems with log fitting.')
            sens_P = np.NaN
            sens_PET = np.NaN
            R2 = np.NaN
            nr_years = len(df_annual)
            VIF = np.NaN
            corr = np.NaN

    return sens_P, sens_PET, R2, nr_years, VIF, corr


def sig_SensitivityBudyko(Q, t, P, PET, wateryear="A-SEP", plot_results=False, fit_intercept=False, use_delta=False):
    """
    ...

    Parameters:
    Q (array-like): Streamflow [mm/timestep]
    t (array-like): Time [datetime or numeric]
    P (array-like): Precipitation [mm/timestep]
    PET (array-like): Potential evapotranspiration [mm/timestep]
    threshold (float, optional): Temperature threshold to distinguish between rain and snow [°C]. Default is 0.

    Returns:

    """
    # Input validation
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("... must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
    df_annual = df_annual.dropna()
    df_annual["Q_Budyko"] = util_TurcPike.calculate_streamflow(df_annual["P"].values, df_annual["PET"].values, 2)

    if len(df_annual) < 10:
        print("Time series shorter than 10y")
        sens_P = np.NaN
        sens_PET = np.NaN
        R2 = np.NaN
        nr_years = len(df_annual)
        VIF = np.NaN
        corr = np.NaN

    else:
        if use_delta:

            X = df_annual[["P", "PET"]] - df_annual[["P", "PET"]].mean()
            y = df_annual["Q_Budyko"] - df_annual["Q_Budyko"].mean()
        else:
            X = df_annual[["P", "PET"]]
            y = df_annual["Q_Budyko"]

        nr_years = len(df_annual)

        mlg_model = LinearRegression(fit_intercept=fit_intercept)
        mlg_model.fit(X, y)

        sens_P = mlg_model.coef_[0]
        sens_PET = mlg_model.coef_[1]
        R2 = mlg_model.score(X, y)

        # calculate VIF for predictors
        if fit_intercept:
            X_c = add_constant(X)
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_c.columns
            VIF = [variance_inflation_factor(X_c.values, i) for i in range(X_c.shape[1])]
        else:
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            VIF = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        corr = np.corrcoef(X["P"], X["PET"])[0, 1]

        # Optional plotting
        if plot_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
            scatter1 = ax1.scatter(X["P"], y, c=X["PET"], s=50, cmap='viridis', alpha=0.8)
            ax1.set_xlabel("P [mm]")
            ax1.set_ylabel("Q Budyko [mm]")
            ax1.set_title("dQ/dP = {:.2f}".format(sens_P))
            cbar1 = fig.colorbar(scatter1, ax=ax1)
            cbar1.set_label("PET [mm]")
            scatter2 = ax2.scatter(X["PET"], y, c=X["P"], s=50, cmap='plasma', alpha=0.8)
            ax2.set_xlabel("PET [mm]")
            ax2.set_ylabel("Q Budyko [mm]")
            ax2.set_title("dQ/dPET = {:.2f}".format(sens_PET))
            cbar2 = fig.colorbar(scatter2, ax=ax2)
            cbar2.set_label("P [mm]")
            plt.show()

    return sens_P, sens_PET, R2, nr_years, VIF, corr

def sig_SensitivityOverTime(Q, t, P, PET, id, window_years=10, wateryear="A-SEP", plot_results=False, fit_intercept=False, use_delta=False):
    """
    Calculate hydrologic metrics using moving water year windows.

    Parameters:
    df : pandas.DataFrame
        Input dataframe with columns: ['date', 'OBS_RUN', 'PRCP', 'PET']
    window_years : int
        Size of the moving window in water years (default=10)
        ...

    Returns:
    pandas.DataFrame with calculated metrics for each water year window
    """

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df['wateryear'] = df["t"].dt.to_period(wateryear)
    water_years = df['wateryear'].unique()
    results = []

    if len(water_years) < window_years:
        print("Not enough water years for the specified window size.")

        # Store results
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
            'nr_years': np.nan,
            'VIF': np.nan
        })

    else:
        # Slide window through the data
        for i in range(len(water_years) - window_years + 1):
            start_year = water_years[i]
            end_year = water_years[i + window_years - 1]

            # Filter data for the current window
            mask = (df['wateryear'] >= start_year) & (df['wateryear'] <= end_year)
            window_df = df[mask]

            # Calculate basic metrics
            mean_P = window_df['P'].mean()
            mean_PET = window_df['PET'].mean()
            mean_Q = window_df['Q'].mean()
            aridity = mean_PET / mean_P if mean_P != 0 else np.nan

            # Calculate sensitivities using function
            sens_P, sens_PET, R2, nr_years, VIF, corr = (
                sig_Sensitivity(window_df["Q"].values, window_df["t"].values, window_df["P"].values, window_df["PET"].values,
                                plot_results=False, use_delta=use_delta, fit_intercept=fit_intercept))

            # Store results
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
                'nr_years': nr_years,
                'VIF': VIF
            })

    df_results = pd.DataFrame(results)

    if plot_results:
        try:
            fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)  # Three rows, one column

            axes[0].plot(df_results['start_wateryear'].dt.year, df_results['mean_P'], label='P', color='tab:blue')
            axes[0].plot(df_results['start_wateryear'].dt.year, df_results['mean_PET'], label='PET', color='tab:orange')
            axes[0].plot(df_results['start_wateryear'].dt.year, df_results['mean_Q'], label='Q', color='tab:purple')
            #axes[0].plot(t, Q, color='tab:purple', alpha=0.2)
            df_results['Q_Budyko'] = util_TurcPike.calculate_streamflow(df_results['mean_P'].values, df_results['mean_PET'].values, 2)
            axes[0].plot(df_results['start_wateryear'].dt.year, df_results['Q_Budyko'], label='Q Budyko', color='tab:purple', linestyle='--')
            axes[0].set_ylabel('Flux [mm/year]')
            axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

            axes[1].plot(df_results['start_wateryear'].dt.year, df_results['sens_P'], label='dQ/dP', color='tab:blue')
            axes[1].plot(df_results['start_wateryear'].dt.year, df_results['sens_PET'], label='dQ/dPET', color='tab:orange')
            # calculate theoretical sensitivities with turc pike method and plot them dashed
            df_results['sens_P_Budyko'], df_results["sens_PET_Budyko"] = util_TurcPike.calculate_sensitivities(df_results['mean_P'].values, df_results['mean_PET'].values, 2)
            axes[1].plot(df_results['start_wateryear'].dt.year, df_results['sens_P_Budyko'], label='dQ/dP Budyko', color='tab:blue', linestyle='--')
            axes[1].plot(df_results['start_wateryear'].dt.year, df_results['sens_PET_Budyko'], label='dQ/dPET Budyko', color='tab:orange', linestyle='--')
            axes[1].axhline(0, color='darkgrey', linestyle='--')
            axes[1].set_xlabel('Water Year')
            axes[1].set_ylabel('Sensitivity [-]')
            axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

            ax3 = axes[2]
            ax3.plot(df_results['start_wateryear'].dt.year, df_results['aridity'], label='PET/P', color='tab:red')
            ax3.set_ylabel('Aridity [-]', color='tab:red')
            ax3.tick_params(axis='y', labelcolor='tab:red')
            ax3_corr = ax3.twinx()
            ax3_corr.plot(df_results['start_wateryear'].dt.year, df_results['P_PET_correlation'], label='Cor(P,PET)',  color='tab:purple')
            ax3_corr.set_ylabel('Correlation [-]', color='tab:purple')
            ax3_corr.tick_params(axis='y', labelcolor='tab:purple')
            ax3.set_xlabel('Water Year')

            fig.tight_layout()
            #plt.show()
            plt.close()
            # save plot
            fig.savefig("D:/Python/Streamflow_sensitivity/figures/temporal/sig_SensitivityOverTime"+str(id)+".png", dpi=300, bbox_inches='tight')
        except:
            print("Plotting didn't work:" + str(id))

    return df_results


def sig_SensitivityWithStorage(Q, t, P, PET, wateryear="A-SEP", plot_results=False, fit_intercept=False, use_delta=False):
    """
    ...

    Parameters:
    Q (array-like): Streamflow [mm/timestep]
    t (array-like): Time [datetime or numeric]
    P (array-like): Precipitation [mm/timestep]
    PET (array-like): Potential evapotranspiration [mm/timestep]
    threshold (float, optional): Temperature threshold to distinguish between rain and snow [°C]. Default is 0.

    Returns:

    """
    # Input validation
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("... must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    fig_handles = {}

    # todo: data check etc.

    df = pd.DataFrame({'t': t, 'Q': Q, 'P': P, 'PET': PET})
    df["wateryear"] = df["t"].dt.to_period(wateryear)
    df_annual = df.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365

    df_annual.loc[:, "Qlag1"] = df_annual["Q"].shift(1)
    df_annual = df_annual.dropna()

    if len(df_annual) < 10:
        print("Time series shorter than 10y")
        # todo: add flag
        sens_P = np.NaN
        sens_PET = np.NaN
        sens_Q = np.NaN
        R2 = np.NaN
        nr_years = len(df_annual)

    else:
        if use_delta:
            X = df_annual[["P", "PET", "Qlag1"]] - df_annual[["P", "PET", "Qlag1"]].mean()
            y = df_annual["Q"] - df_annual["Q"].mean()
        else:
            X = df_annual[["P", "PET", "Qlag1"]]
            y = df_annual["Q"]

        mlg_model = LinearRegression(fit_intercept=fit_intercept)
        mlg_model.fit(X, y)

        sens_P = mlg_model.coef_[0]
        sens_PET = mlg_model.coef_[1]
        sens_Q = mlg_model.coef_[2]
        R2 = mlg_model.score(X, y)
        nr_years = len(df_annual)

        # Optional plotting
        if plot_results:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3), tight_layout=True)
            scatter1 = ax1.scatter(X["P"], y, c=X["PET"], s=X["Qlag1"]*50, cmap='viridis', alpha=0.8)
            ax1.set_xlabel("P [mm]")
            ax1.set_ylabel("Q [mm]")
            ax1.set_title("dQ/dP = {:.2f}".format(sens_P))
            cbar1 = fig.colorbar(scatter1, ax=ax1)
            cbar1.set_label("PET [mm]")
            scatter2 = ax2.scatter(X["PET"], y, c=X["P"], s=X["Qlag1"]*50, cmap='plasma', alpha=0.8)
            ax2.set_xlabel("PET [mm]")
            ax2.set_ylabel("Q [mm]")
            ax2.set_title("dQ/dPET = {:.2f}".format(sens_PET))
            cbar2 = fig.colorbar(scatter2, ax=ax2)
            cbar2.set_label("P [mm]")
            scatter3 = ax3.scatter(X["Qlag1"], y, s=50, alpha=0.8)
            ax3.set_xlabel("Qlag1 [mm]")
            ax3.set_ylabel("Q [mm]")
            ax3.set_title("dQ/dQlag1 = {:.2f}".format(sens_Q))
            plt.show()

    return sens_P, sens_PET, sens_Q, R2, nr_years

