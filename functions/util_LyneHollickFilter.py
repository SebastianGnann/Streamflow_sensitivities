import numpy as np


def util_LyneHollickFilter(Q, filter_parameter=0.925, nr_passes=1, threshold_type='pass'):
    """
    Estimates baseflow using the Lyne-Hollick filter.

    Parameters:
        Q (array_like): Streamflow data [mm/timestep].
        filter_parameter (float, optional): Filter parameter (0 to 1). Default is 0.925.
        nr_passes (int, optional): Number of passes. Default is 1.
        threshold_type (str, optional): How to threshold resulting time series ('end', 'pass', 'timestep', 'none'). Default is 'pass'.

    Returns:
        np.ndarray: Baseflow [mm/timestep].
    """

    Q = np.asarray(Q).flatten()

    if not 0 <= filter_parameter <= 1:
        raise ValueError("Filter parameter must be between 0 and 1.")

    if not isinstance(nr_passes, int) or nr_passes < 1:
        raise ValueError(
            "Number of filter passes must be an integer greater than zero.")  # also needs to be an odd number!

    threshold_type = threshold_type.lower()
    if threshold_type not in ['end', 'timestep', 'pass', 'none']:
        raise ValueError("Not a valid thresholding method. Choose either end, timestep, pass, or none.")

    # Handle NaN values by replacing them with the median
    Q_tmp = Q.copy()
    nan_mask = np.isnan(Q)
    Q_tmp[nan_mask] = np.nanmedian(Q)

    Q_b = LyneHollickFilter(Q_tmp, filter_parameter, threshold_type)
    for _ in range(1, nr_passes):
        Q_b = LyneHollickFilter(np.flip(Q_b), filter_parameter, threshold_type)
        Q_b = np.flip(Q_b)  # Flip back to original direction

    Q_b[nan_mask] = np.nan  # Set baseflow to NaN where streamflow is NaN

    # constrain baseflow not to be higher than streamflow
    if threshold_type != 'none':
        Q_b = np.minimum(Q_b, Q)

    return Q_b


def LyneHollickFilter(Q, filter_parameter, threshold_type):
    """Helper function that runs the Lyne-Hollick filter."""
    n = len(Q)
    Q_f = np.full(n, np.nan)
    Q_f[0] = Q[0] - np.min(Q)  # Initial condition, see Su et al. (2016)

    threshold_timestep = False
    threshold_pass = False

    if threshold_type == 'end':
        pass
    elif threshold_type == 'timestep':
        threshold_timestep = True
    elif threshold_type == 'pass':
        threshold_pass = True
    elif threshold_type == 'none':
        pass
    else:
        raise ValueError('Not a valid thresholding method. Choose either end, timestep, pass, or none.')

    if threshold_timestep:
        for i in range(1, n):
            Q_f[i] = filter_parameter * Q_f[i - 1] + ((1 + filter_parameter) / 2) * (Q[i] - Q[i - 1])
            if Q_f[i] < 0:  # Constrain after each timestep
                Q_f[i] = 0
    else:
        for i in range(1, n):
            Q_f[i] = filter_parameter * Q_f[i - 1] + ((1 + filter_parameter) / 2) * (Q[i] - Q[i - 1])

    if threshold_pass:
        Q_f[Q_f < 0] = 0  # Constrain after each filter pass

    # Calculate baseflow
    Q_b = Q - Q_f

    return Q_b
