import numpy as np
import pandas as pd

def calculate_seasonality_index(P, PET, dates):
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
