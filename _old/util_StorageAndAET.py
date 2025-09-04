import numpy as np
from datetime import datetime


def util_StorageAndAET(Q, t, P, PET, field_capacity=None):
    """
    Calculates storage and actual evapotranspiration using a simple soil moisture model.

    Parameters:
    Q (array-like): Streamflow [mm/timestep]
    t (array-like): Time [datetime or numeric]
    P (array-like): Precipitation [mm/timestep]
    PET (array-like): Potential evapotranspiration [mm/timestep]
    field_capacity (float, optional): Field capacity [mm]

    Returns:
    tuple: (S, AET) where S is storage [mm] and AET is actual evapotranspiration [mm/timestep]
    """
    # Input validation
    if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in [Q, P, PET]):
        raise ValueError("Q, P, and PET must be 1D numpy arrays")

    if not isinstance(t, (np.ndarray, list)) or (
            isinstance(t[0], (int, float, np.number)) and not isinstance(t[0], datetime)):
        raise ValueError("t must be a list or numpy array of datetime or numeric values")

    if len(Q) != len(t) or len(P) != len(t) or len(PET) != len(t):
        raise ValueError("All input arrays must have the same length")

    # Initialize arrays
    AET = np.full_like(Q, np.nan)
    S = np.full_like(Q, np.nan)
    a = np.full_like(Q, np.nan)

    # Estimate field capacity if not provided
    if field_capacity is None:
        field_capacity = 200  # initial guess
        AET[0] = PET[0]
        S[0] = field_capacity
        a[0] = 1

        for i in range(1, len(t)):
            a[i] = S[i - 1] / field_capacity if S[i - 1] < field_capacity else 1
            AET[i] = a[i] * PET[i]
            S[i] = P[i] - Q[i] - AET[i] + S[i - 1]
            if S[i] < 0:
                AET[i] = P[i] - Q[i] + S[i - 1]
                S[i] = 0
            if AET[i] < 0:
                AET[i] = 0

        field_capacity = np.max(S)

    # Run loop again with provided or estimated field capacity
    S[0] = field_capacity

    for i in range(1, len(t)):
        a[i] = S[i - 1] / field_capacity if S[i - 1] < field_capacity else 1
        AET[i] = a[i] * PET[i]
        S[i] = P[i] - Q[i] - AET[i] + S[i - 1]
        if S[i] < 0:
            AET[i] = P[i] - Q[i] + S[i - 1]
            S[i] = 0
        if AET[i] < 0:
            AET[i] = 0
    return S, AET
