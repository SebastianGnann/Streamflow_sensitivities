import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions.util_DataCheck import util_DataCheck
from functions.util_StorageAndAET import util_StorageAndAET
from datetime import datetime


def sig_StorageWaterBalance(Q, t, P, PET, field_capacity=None, plot_results=False):
    """
    Calculate ratio between active and total storage.

    Parameters:
    Q (array-like): Streamflow [mm/timestep]
    t (array-like): Time (datetime objects)
    P (array-like): Precipitation [mm/timestep]
    PET (array-like): Potential evapotranspiration [mm/timestep]
    field_capacity (float, optional): Field capacity [mm]
    plot_results (bool, optional): Whether to plot results (default: False)

    Returns:
    tuple: (S_..., error_flag, error_str, fig_handles)
    """
    # Input validation
    if not all(len(arr) == len(Q) for arr in [t, P, PET]):
        raise ValueError("All input arrays must have the same length.")

    # Initialize output variables
    error_flag = 0
    error_str = ""
    fig_handles = {}

    # Data checks (assuming util_DataCheck function exists)
    error_flag, error_str, timestep, t = util_DataCheck(Q, t, P=P, PET=PET)
    if error_flag == 2:
        return np.nan, np.nan, error_flag, error_str, fig_handles

    # Get rid of NaN values (temporarily)
    isn = np.isnan(Q) | np.isnan(P) | np.isnan(PET)
    Q_tmp = Q.copy()
    P_tmp = P.copy()
    PET_tmp = PET.copy()
    # overwrites values...
    Q_tmp[isn] = np.nanmean(Q_tmp)
    P_tmp[isn] = np.nanmean(P_tmp)
    PET_tmp[isn] = np.nanmean(PET_tmp)

    # Estimate storage (assuming util_StorageAndAET function exists)
    S, AET = util_StorageAndAET(Q_tmp, t, P_tmp, PET_tmp, field_capacity=field_capacity)

    #Q_tmp[isn] = np.nan
    S[isn] = np.nan
    S_range = np.nanmax(S) - np.nanmin(S)

    S_annual_range = pd.Series(S, index=t).resample('A-SEP').max() - pd.Series(S, index=t).resample('A-SEP').min()
    S_annual_range_median = S_annual_range.median() # more robust to trends

    # Optional plotting
    if plot_results:
        fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
        im = ax.plot(t, P, c='grey', alpha=0.8, lw=1, label='P')
        im = ax.plot(t, Q, c='tab:blue', alpha=0.8, lw=1, label='Q')
        im = ax.plot(t, PET, c='tab:red', alpha=0.8, lw=1, label='PET')
        im = ax.plot(t, AET, c='tab:orange', alpha=0.8, lw=1, label='AET')
        ax.set_xlabel('Time')
        plt.xlim(datetime(2000, 10, 1), datetime(2009, 9, 30))
        ax.set_ylabel('Fluxes [mm/timestep]')
        ax2 = ax.twinx()
        im = ax2.plot(t, S, c='tab:purple', alpha=0.8, lw=1, label='S')
        ax2.set_ylabel('Storage [mm]')
        ax.legend()

        plt.show()

    return S_range, S_annual_range_median, error_flag, error_str, fig_handles
