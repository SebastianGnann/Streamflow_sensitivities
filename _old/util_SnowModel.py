import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_swe(prec, date, temp, cfmax=4.5, tt=0.0, plot_results=False):
    """
    Calculate Snow Water Equivalent using a degree-day approach with date handling

    Parameters:
    date (pd.Series or np.ndarray): Date series
    prec (pd.Series or np.ndarray): Daily precipitation (mm/day)
    temp (pd.Series or np.ndarray): Daily temperature (°C)
    cfmax (float): Degree-day factor (mm/°C/day)
    tt (float): Threshold temperature (°C)
    plot_results (bool): Show interactive plot if True

    Returns:
    pd.DataFrame: Contains three columns:
        - 'melt': Melt water (mm/day)
        - 'rain': Rain (mm/day)
        - 'swe': Snow pack (mm)
    """

    # todo: pM,ossible additions
    # - account for hypsometry
    # - vary parameters throughout the year

    # Ensure inputs are pandas Series
    if isinstance(prec, np.ndarray):
        prec = pd.Series(prec)
    if isinstance(temp, np.ndarray):
        temp = pd.Series(temp)
    if isinstance(date, np.ndarray):
        date = pd.Series(date)

    # Create DataFrame with dates
    df = pd.DataFrame({
        'date': date,
        'prec': prec.fillna(0.0),  # Fill missing precipitation with 0
        'temp': temp.fillna(5.0)   # Fill missing temperature with 5°C
    })

    # Temperature masks
    cold_mask = df['temp'] <= tt
    warm_mask = ~cold_mask

    # Accumulation and melt calculations
    df['accu'] = np.where(cold_mask, df['prec'], 0.0)
    df['psm'] = np.where(warm_mask, cfmax * (df['temp'] - tt), 0.0)
    df['rain'] = np.where(warm_mask, df['prec'], 0.0)

    # Vectorized snowpack simulation
    swe = np.zeros(len(df))
    melt = np.zeros(len(df))

    for i in range(1, len(df)):
        swe[i] = swe[i - 1] + df['accu'].iloc[i]
        melt[i] = min(swe[i], df['psm'].iloc[i])
        swe[i] -= melt[i]

    df['swe'] = swe
    df['melt'] = melt

    # Apply original NaN masks
    na_mask = temp.isna() | prec.isna()
    df[['melt', 'rain', 'swe']] = df[['melt', 'rain', 'swe']].mask(na_mask)

    # Plotting functionality
    if plot_results:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df['date'], df['prec'], label='Precipitation (mm/day)', alpha=0.8, c='grey')
        ax.plot(df['date'], df['melt'], label='Meltwater (mm/day)', alpha=0.8, c='tab:blue')
        ax.set_ylabel('Flux (mm)')
        ax.set_xlabel('Date')
        plt.legend()
        ax2 = plt.twinx()
        ax2.plot(df['date'], df['swe'], label='Snowpack (mm)', alpha=0.8, c='tab:orange')
        ax2.set_ylabel('SWE (mm)')
        plt.tight_layout()
        plt.show()

    return df[['melt', 'rain', 'swe']]
