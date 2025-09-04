import numpy as np
from datetime import datetime, timedelta


def util_DataCheck(Q, t, **kwargs):
    """
    Check data for various issues.

    Parameters:
    Q (array-like): Streamflow [mm/timestep]
    t (array-like): Time (datetime objects or numeric)
    P (array-like, optional): Precipitation [mm/timestep]
    PET (array-like, optional): Potential evapotranspiration [mm/timestep]
    T (array-like, optional): Temperature [degC]

    Returns:
    tuple: (error_flag, error_str, timestep, t)
    """
    # Input validation
    if not (isinstance(Q, (list, np.ndarray)) and isinstance(t, (list, np.ndarray))):
        raise ValueError("Q and t must be array-like.")

    # Optional inputs
    P = kwargs.get('P', None)
    PET = kwargs.get('PET', None)
    T = kwargs.get('T', None)

    # Initialize output variables
    error_flag = 0
    error_str = ""

    # Timestep checks
    if np.issubdtype(type(t[0]), np.number):
        error_flag = 1
        t = [datetime.fromordinal(int(d)) for d in t]
        error_str += "Warning: Converted numeric to datetime. "

    timesteps = np.diff(t).astype('timedelta64[s]').astype(np.int64)
    timestep = np.median(timesteps)
    if not np.all(np.diff(timesteps) == 0):
        error_flag = 1
        error_str += "Warning: Record is not continuous (some timesteps are missing). "

    # Data checks
    if np.nanmin(Q) < 0:
        error_flag = 2
        error_str += "Error: Negative values in flow series. "
        return error_flag, error_str, timestep, t

    if np.all(Q == 0):
        error_flag = 2
        error_str += "Error: Only zero flow in flow series. "
        return error_flag, error_str, timestep, t

    if len(Q) != len(t):
        error_flag = 2
        error_str += "Error: Flow series and time vector have different lengths. "
        return error_flag, error_str, timestep, t

    if np.any(np.isnan(Q)):
        error_flag = 1
        error_str += "Warning: Ignoring NaNs in streamflow data. "

    if np.all(np.isnan(Q)):
        error_flag = 2
        error_str += "Error: Only NaNs in streamflow data. "
        return error_flag, error_str, timestep, t

    if len(Q) < 30:
        error_flag = 1
        error_str += "Warning: Extremely short time series. "

    # Check P if provided
    if P is not None:
        if np.any(np.isnan(P)):
            error_flag = 1
            error_str += "Warning: Ignoring NaNs in precipitation data. "

        if np.all(np.isnan(P)):
            error_flag = 2
            error_str += "Error: Only NaNs in precipitation data. "
            return error_flag, error_str, timestep, t

        if len(Q) != len(P):
            error_flag = 2
            error_str += "Error: Precipitation and flow series have different lengths. "
            return error_flag, error_str, timestep, t

        if np.nanmin(P) < 0:
            error_flag = 2
            error_str += "Error: Negative values in precipitation series. "
            return error_flag, error_str, timestep, t

    # Check PET if provided
    if PET is not None:
        if np.any(np.isnan(PET)):
            error_flag = 1
            error_str += "Warning: Ignoring NaNs in potential evapotranspiration data. "

        if np.all(np.isnan(PET)):
            error_flag = 2
            error_str += "Error: Only NaNs in potential evapotranspiration data. "
            return error_flag, error_str, timestep, t

        if len(Q) != len(PET):
            error_flag = 2
            error_str += "Error: Potential evapotranspiration and flow series have different lengths. "
            return error_flag, error_str, timestep, t

        if np.nanmin(PET) < 0:
            error_flag = 1
            error_str += "Warning: Negative values in potential evapotranspiration series. "

    # Check T if provided
    if T is not None:
        if np.any(np.isnan(T)):
            error_flag = 1
            error_str += "Warning: Ignoring NaNs in temperature data. "

        if np.all(np.isnan(T)):
            error_flag = 2
            error_str += "Error: Only NaNs in temperature data. "
            return error_flag, error_str, timestep, t

        if len(Q) != len(T):
            error_flag = 2
            error_str += "Error: Temperature and flow series have different lengths. "
            return error_flag, error_str, timestep, t

        if np.nanmin(T) < -273.15:
            error_flag = 2
            error_str += "Error: Temperature cannot be less than the absolute minimum of -273.15 degC. "
            return error_flag, error_str, timestep, t

    return error_flag, error_str, timedelta(seconds=timestep), t
