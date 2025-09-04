import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def calculate_Woods_seasonality_index(P, T, dates):
    """
    Calculate the Woods seasonality index (P*) based on daily precipitation (P)
    and temperature (T) data. Based on code by Nans Addor.

    Parameters:
        P (array-like): Daily precipitation values.
        T (array-like): Daily temperature values.
        dates (array-like): Corresponding dates for P and T (datetime format).

    Returns:
        float: Seasonality index (P*).
    """
    # Prepare DataFrame and drop rows with missing values in either variable
    data = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Precipitation": P,
        "Temperature": T
    })

    # check if any in P or T is NaN
    if data["Precipitation"].isnull().any() or data["Temperature"].isnull().any():
        #print("Precipitation or temperature series contain NaN values, seasonality cannot be calculated.")
        P_star = np.nan

    else:
        t_julian = data["Date"].dt.dayofyear.values
        prec = data["Precipitation"].values
        temp = data["Temperature"].values

        # Estimate initial phase for precipitation (as in R code)
        data['Month'] = data['Date'].dt.month
        prec_monthly = data.groupby('Month')['Precipitation'].mean()
        s_p_guess = (90 - prec_monthly.idxmax() * 30) % 360

        # Sine curve fitting functions
        def temp_sine(t, delta_t, s_t):
            return np.mean(temp) + delta_t * np.sin(2 * np.pi * (t - s_t) / 365.25)
        def prec_sine(t, delta_p, s_p):
            return np.mean(prec) * (1 + delta_p * np.sin(2 * np.pi * (t - s_p) / 365.25))

        # Fit temperature sine
        popt_temp, _ = curve_fit(temp_sine, t_julian, temp, p0=[5, -90], maxfev=10000)
        delta_t, s_t = popt_temp

        # Fit precipitation sine
        popt_prec, _ = curve_fit(prec_sine, t_julian, prec, p0=[0.4, s_p_guess], maxfev=10000)
        delta_p, s_p = popt_prec

        # Seasonality index (P*)
        P_star = delta_p * np.sign(delta_t) * np.cos(2 * np.pi * (s_p - s_t) / 365.25)

    return P_star

def calculate_Knoben_seasonality_index(P, PET, dates):
    """
    Calculate the seasonality index (Im,r) based on daily precipitation (P)
    and potential evapotranspiration (PET) data.

    Parameters:
        P (array-like): Daily precipitation values.
        PET (array-like): Daily potential evapotranspiration values.
        dates (array-like): Corresponding dates for P and PET (datetime format).

    Returns:
        float: Seasonality index (Im,r).
    """
    # Step 1: Create a DataFrame from inputs
    data = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Precipitation": P,
        "PET": PET
    })

    # Step 2: Aggregate daily data into monthly totals
    data["Month"] = data["Date"].dt.month
    monthly_data = data.groupby("Month").agg({
        "Precipitation": "sum",  # Sum daily precipitation to get monthly totals
        "PET": "sum"             # Sum daily PET to get monthly totals
    }).reset_index()

    # Step 3: Calculate monthly moisture index (Im)
    def moisture_index(P, PET):
        return (P - PET) / np.maximum(P, PET)

    monthly_data["Moisture_Index"] = moisture_index(monthly_data["Precipitation"], monthly_data["PET"])

    # Step 4: Compute seasonality index (Im,r)
    Im_values = monthly_data["Moisture_Index"]
    Im_r = np.max(Im_values) - np.min(Im_values)

    return Im_r
