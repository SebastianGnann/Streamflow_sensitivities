import numpy as np
from scipy import interpolate


def util_UKIH_Method(Q, n_days=5):
    """
    Estimates baseflow with UKIH method.

    Parameters:
    Q (array-like): Streamflow [mm/timestep]
    n_days (int, optional): Length of data blocks. Default is 5 days.

    Returns:
    numpy.ndarray: Baseflow [mm/timestep]
    """
    if not isinstance(Q, (np.ndarray, list)):
        raise ValueError("Q must be a numpy array or a list")

    Q = np.array(Q).flatten()

    if not isinstance(n_days, int) or n_days < 1:
        raise ValueError("Filter window must be an integer larger than zero.")

    # Handle NaN values
    Q_tmp = Q.copy()
    nan_mask = np.isnan(Q)
    Q_tmp[nan_mask] = np.nanmedian(Q)

    # Calculate baseflow
    Q_b, t_ind = UKIH_Method(Q_tmp, n_days)

    # Use minimum baseflow to fill in missing values at the beginning
    B_tmp = np.full_like(Q_tmp, np.nanmin(Q_tmp))
    B_tmp[t_ind] = Q_b
    Q_b = B_tmp

    # Set baseflow to NaN where streamflow is NaN
    Q_b[nan_mask] = np.nan

    return Q_b


def UKIH_Method(Q, n_days):
    """Helper function that runs the UKIH method."""
    n = len(Q)
    Q_min5 = np.full(round(n / n_days), np.nan)
    min_i = np.full(round(n / n_days), np.nan)
    ind5 = 0
    TP = []
    t_TP = []

    for i in range(0, n - n_days + 1, n_days):
        Q_min5[ind5], min_i[ind5] = min((Q[i + j], j) for j in range(n_days))
        min_i[ind5] += i
        if ind5 >= 2:
            if (Q_min5[ind5 - 1] * 0.9 < Q_min5[ind5 - 2] and
                    Q_min5[ind5 - 1] * 0.9 < Q_min5[ind5]):
                TP.append(Q_min5[ind5 - 1])
                t_TP.append(int(min_i[ind5 - 1]))
        ind5 += 1

    if not t_TP:
        t_ind = np.arange(len(Q))
        Q_b = np.full(len(Q), np.nan)
    else:
        t_ind = np.arange(t_TP[0], t_TP[-1] + 1)
        f = interpolate.interp1d(t_TP, TP, kind='linear', fill_value='extrapolate')
        Q_b = f(t_ind)
        Qt = Q[t_ind]
        Q_b[Q_b > Qt] = Qt[Q_b > Qt]

    return Q_b, t_ind

    '''
    # baseflow plot
    start_date = datetime(2008, 10, 1)
    end_date = datetime(2010, 9, 30)
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    im = ax.plot(df_tmp["date"], Q_tot, c='grey', alpha=0.8, lw=2)
    im = ax.plot(df_tmp["date"], Q_b5, c='tab:orange', alpha=0.8, lw=2)
    im = ax.plot(df_tmp["date"], Q_b90, c='tab:purple', alpha=0.8, lw=2)
    plt.xlim(start_date, end_date)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Major ticks every 3 months
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format date as 'Mon YYYY'
    plt.show()
    '''
