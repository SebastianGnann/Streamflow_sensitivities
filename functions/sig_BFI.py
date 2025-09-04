import numpy as np
import matplotlib.pyplot as plt
from functions.util_DataCheck import util_DataCheck
from functions.util_LyneHollickFilter import util_LyneHollickFilter
from functions.util_UKIH_Method import util_UKIH_Method


def sig_BFI(Q, t, **kwargs):
    """
    Calculate baseflow index (BFI).

    Parameters:
    Q (array-like): Streamflow [mm/timestep]
    t (array-like): Time (datetime objects)
    method (str, optional): Baseflow separation method ('Lyne_Hollick' or 'UKIH'). Default is 'UKIH'.
    parameters (list, optional): Filter parameters. Default is None.
    plot_results (bool, optional): Whether to plot results. Default is False.

    Returns:
    tuple: (BFI, error_flag, error_str, fig_handles)
    """
    # Input validation
    if len(Q) != len(t):
        raise ValueError("Q and t must have the same length.")

    # todo: check if parameters are in correct format

    # Default values
    method = kwargs.get('method', 'UKIH')
    parameters = kwargs.get('parameters', None)
    plot_results = kwargs.get('plot_results', False)

    # Initialize output variables
    error_flag = 0
    error_str = ""
    fig_handles = {}

    # Data checks (assuming util_DataCheck function exists)
    error_flag, error_str, timestep, t = util_DataCheck(Q, t)
    if error_flag == 2:
        return np.nan, error_flag, error_str, fig_handles

    timestep_factor = 1  # / timestep.days

    # Pad time series
    if len(Q) > 60:
        Q_padded = np.concatenate([Q[29::-1], Q, Q[-30:]])
    else:
        Q_padded = Q
        error_flag = 1
        error_str += "Warning: Very short time series. Baseflow separation might be unreliable. "

    # Obtain baseflow
    if method == 'Lyne_Hollick':
        if parameters is None:
            filter_parameter = np.exp(np.log(0.925) / timestep_factor)
            parameters = [filter_parameter, 3]
        elif len(parameters) == 1:
            parameters = [np.exp(np.log(parameters[0]) / timestep_factor), 3]
        elif len(parameters) == 2:
            parameters[0] = np.exp(np.log(parameters[0]) / timestep_factor)
        else:
            raise ValueError("Too many filter parameters.")

        Q_b = util_LyneHollickFilter(Q_padded, filter_parameter=parameters[0], nr_passes=parameters[1])

    elif method == 'UKIH':
        if parameters is None:
            parameters = [5 * timestep_factor]
        elif len(parameters) == 1:
            parameters[0] *= timestep_factor
        else:
            raise ValueError("Too many filter parameters.")

        Q_b = util_UKIH_Method(Q_padded, n_days=parameters[0])

    else:
        raise ValueError("Please choose one of the available baseflow separation methods (UKIH or Lyne_Hollick).")

    # Remove padding
    if len(Q) > 60:
        Q_b = Q_b[30:-30]

    # Calculate BFI
    BFI = np.nansum(Q_b) / np.nansum(Q)

    # Check if 0 <= BFI <= 1
    if BFI < 0 or BFI > 1:
        BFI = np.nan
        error_flag = 1
        error_str += "Warning: Estimated BFI outside allowed range (0 to 1)."

    # Optional plotting
    if plot_results:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, Q, label='Streamflow')
        ax.plot(t, Q_b, label='Estimated baseflow')
        ax.set_xlabel('Date')
        ax.set_ylabel('Flow [mm/timestep]')
        ax.legend()
        # fig_handles['BFI'] = fig
        plt.show()

    return BFI, error_flag, error_str, fig_handles
