import numpy as np
import pandas as pd
import functions.util_SnowModel as util_SnowModel
from functions.util_Seasonality import calculate_Knoben_seasonality_index
from functions.util_Seasonality import calculate_Woods_seasonality_index
from functions.sig_Sensitivity import sig_Sensitivity, sig_SensitivityLog, sig_SensitivityWithStorage, sig_SensitivityOverTime, sig_SensitivityAveraging
from functions.util_WaterYear import adjust_to_water_year_month, get_water_year_string
from functions.sig_BFI import sig_BFI

def initialize_result_lists():
    return {
        "gauge_id_native": [],
        "gauge_id": [],
        "perc_complete": [],
        "neg_count": [],
        "len_years": [],
        "frac_snow_control": [],
        "aridity_control": [],
        "seasonality_index": [],
        "P_seasonality_index": [],
        "mean_P": [],
        "mean_PET": [],
        "mean_Q": [],
        "mean_T": [],
        "annual_P": [],
        "annual_PET": [],
        "annual_Q": [],
        "annual_T": [],
        "cor_PET_P": [],
        "sensitivity_len": [],
        "sens_P_mr1": [],
        "sens_PET_mr1": [],
        "pval_sens_P_mr1": [],
        "pval_sens_PET_mr1": [],
        "R2_mr1": [],
        "sens_P_mr2": [],
        "sens_PET_mr2": [],
        "pval_sens_P_mr2": [],
        "pval_sens_PET_mr2": [],
        "R2_mr2": [],
        "sens_P_log": [],
        "sens_PET_log": [],
        "pval_sens_P_log": [],
        "pval_sens_PET_log": [],
        "R2_log": [],
        "sens_P_storage_mr1": [],
        "sens_PET_storage_mr1": [],
        "sens_Q_storage_mr1": [],
        "pval_sens_P_storage_mr1": [],
        "pval_sens_PET_storage_mr1": [],
        "pval_sens_Q_storage_mr1": [],
        "R2_storage_mr1": [],
        "sens_P_storage_mr2": [],
        "sens_PET_storage_mr2": [],
        "sens_Q_storage_mr2": [],
        "pval_sens_P_storage_mr2": [],
        "pval_sens_PET_storage_mr2": [],
        "pval_sens_Q_storage_mr2": [],
        "R2_storage_mr2": [],
        "sens_P_avg_mr1": [],
        "sens_PET_avg_mr1": [],
        "pval_sens_P_avg_mr1": [],
        "pval_sens_PET_avg_mr1": [],
        "R2_avg_mr1": [],
        "sens_P_avg_mr2": [],
        "sens_PET_avg_mr2": [],
        "pval_sens_P_avg_mr2": [],
        "pval_sens_PET_avg_mr2": [],
        "R2_avg_mr2": [],
        "start_wateryear": [],
        "end_wateryear": [],
        "mean_P_over_time": [],
        "mean_PET_over_time": [],
        "mean_Q_over_time": [],
        "aridity_over_time": [],
        "cor_PET_P_over_time": [],
        "sens_P_over_time_mr1": [],
        "sens_PET_over_time_mr1": [],
        "R2_over_time_mr1": [],
        "sens_P_over_time_mr2": [],
        "sens_PET_over_time_mr2": [],
        "R2_over_time_mr2": [],
        "BFI": []
    }

def append_results(result_lists, gauge_results):
    for key, value in gauge_results.items():
        if key in result_lists:
            result_lists[key].append(value)
        else:
            # If a new key is found, initialize its list
            result_lists[key] = [value]

def calculate_metrics(df_tmp, id, gauge_id, wy):

    wy_str = get_water_year_string(wy)

    # remove NaNs at beginning and end
    first_valid_index = df_tmp["Q"].first_valid_index()
    df_tmp = df_tmp.loc[first_valid_index:]
    last_valid_index = df_tmp["Q"].last_valid_index()
    df_tmp = df_tmp.loc[:last_valid_index]
    df_tmp = df_tmp.reset_index(drop=True)

    # calculate time series summary metrics
    neg_count = np.sum(df_tmp["Q"].values < 0)
    perc_complete = np.sum(~np.isnan(df_tmp["Q"].values)) / len(df_tmp["Q"].values)
    len_years = (df_tmp["date"].max() - df_tmp["date"].min()).days / 365.25

    if len(df_tmp["date"].values) < 365:
        print("Less than a year of values in time series. Signatures not calculated.")
        frac_snow_control = np.nan
        aridity_control = np.nan
        seasonality_index = np.nan
        P_seasonality_index = np.nan
        mean_P = np.nan
        mean_PET = np.nan
        mean_Q = np.nan
        mean_T = np.nan
        cor_PET_P = np.nan
        annual_P = np.nan
        annual_PET = np.nan
        annual_Q = np.nan
        annual_T = np.nan
        sensitivity_len = np.nan
        sens_P_mr1 = np.nan
        sens_PET_mr1 = np.nan
        pval_sens_P_mr1 = np.nan
        pval_sens_PET_mr1 = np.nan
        R2_mr1 = np.nan
        sens_P_mr2 = np.nan
        sens_PET_mr2 = np.nan
        pval_sens_P_mr2 = np.nan
        pval_sens_PET_mr2 = np.nan
        R2_mr2 = np.nan
        sens_P_log = np.nan
        sens_PET_log = np.nan
        pval_sens_P_log = np.nan
        pval_sens_PET_log = np.nan
        R2_log = np.nan
        sens_P_storage_mr1 = np.nan
        sens_PET_storage_mr1 = np.nan
        sens_Q_storage_mr1 = np.nan
        pval_sens_P_storage_mr1 = np.nan
        pval_sens_PET_storage_mr1 = np.nan
        pval_sens_Q_storage_mr1 = np.nan
        R2_storage_mr1 = np.nan
        sens_P_storage_mr2 = np.nan
        sens_PET_storage_mr2 = np.nan
        sens_Q_storage_mr2 = np.nan
        pval_sens_P_storage_mr2 = np.nan
        pval_sens_PET_storage_mr2 = np.nan
        pval_sens_Q_storage_mr2 = np.nan
        R2_storage_mr2 = np.nan
        sens_P_avg_mr1 = np.nan
        sens_PET_avg_mr1 = np.nan
        pval_sens_P_avg_mr1 = np.nan
        pval_sens_PET_avg_mr1 = np.nan
        R2_avg_mr1 = np.nan
        sens_P_avg_mr2 = np.nan
        sens_PET_avg_mr2 = np.nan
        pval_sens_P_avg_mr2 = np.nan
        pval_sens_PET_avg_mr2 = np.nan
        R2_avg_mr2 = np.nan
        df_time_mr1 = pd.DataFrame({
            "start_wateryear": [],
            "end_wateryear": [],
            "mean_P": [],
            "mean_PET": [],
            "mean_Q": [],
            "aridity": [],
            "P_PET_correlation": [],
            "sens_P": [],
            "sens_PET": [],
            "pval_sens_P": [],
            "pval_sens_PET": [],
            "R2": []
        })
        df_time_mr2 = pd.DataFrame({
            "sens_P": [],
            "sens_PET": [],
            "pval_sens_P": [],
            "pval_sens_PET": [],
            "R2": []
        })
        BFI = np.nan

    else:
        # calculate SWE and snow fraction
        df_snow = util_SnowModel.calculate_swe(df_tmp["P"].values, df_tmp["date"].values, df_tmp["T"].values,
                                               plot_results=False)
        df_tmp["melt"] = df_snow["melt"]
        df_tmp["rain"] = df_snow["rain"]
        df_tmp["swe"] = df_snow["swe"]

        frac_snow_control = np.nanmean(df_tmp["melt"]) / np.nanmean(df_tmp["P"])

        # calculate mean values, aridity and seasonality
        seasonality_index = calculate_Knoben_seasonality_index(
            df_tmp["P"].values, df_tmp["PET"].values, df_tmp["date"].values)

        P_seasonality_index = calculate_Woods_seasonality_index(
            df_tmp["P"].values, df_tmp["T"].values, df_tmp["date"].values)

        mean_P = np.nanmean(df_tmp["P"].values)
        mean_PET = np.nanmean(df_tmp["PET"].values)
        mean_Q = np.nanmean(df_tmp["Q"].values)
        mean_T = np.nanmean(df_tmp["T"].values)

        aridity_control = mean_PET/mean_P

        # calculate annual means
        df_tmp["wateryear"] = df_tmp["date"].dt.to_period(wy_str)
        df_annual = df_tmp.groupby("wateryear").sum(min_count=365, numeric_only=True) / 365
        df_annual = df_annual.dropna()
        annual_P = df_annual["P"].values
        annual_PET = df_annual["PET"].values
        annual_Q = df_annual["Q"].values
        annual_T = df_annual["T"].values

        # sensitivities
        sens_P_mr1, sens_PET_mr1, R2_mr1, sensitivity_len, VIF, cor_PET_P, pval_sens_P_mr1, pval_sens_PET_mr1 = sig_Sensitivity(
            df_tmp["Q"].values, df_tmp["date"].values, df_tmp["P"].values,
            df_tmp["PET"].values, use_delta=False, fit_intercept=False, wateryear=wy_str)
        sens_P_mr2, sens_PET_mr2, R2_mr2, _, _, _, pval_sens_P_mr2, pval_sens_PET_mr2 = sig_Sensitivity(
            df_tmp["Q"].values, df_tmp["date"].values, df_tmp["P"].values,
            df_tmp["PET"].values, use_delta=True, fit_intercept=True, wateryear=wy_str)
        sens_P_log, sens_PET_log, R2_log, _, _, _, pval_sens_P_log, pval_sens_PET_log = sig_SensitivityLog(
            df_tmp["Q"].values, df_tmp["date"].values, df_tmp["P"].values,
            df_tmp["PET"].values, use_delta=False, fit_intercept=True, wateryear=wy_str)
        sens_P_storage_mr1, sens_PET_storage_mr1, sens_Q_storage_mr1, R2_storage_mr1, _, pval_sens_P_storage_mr1, pval_sens_PET_storage_mr1, pval_sens_Q_storage_mr1 \
            = sig_SensitivityWithStorage(
            df_tmp["Q"].values, df_tmp["date"].values, df_tmp["P"].values,
            df_tmp["PET"].values, use_delta=False, fit_intercept=False, wateryear=wy_str)
        sens_P_storage_mr2, sens_PET_storage_mr2, sens_Q_storage_mr2, R2_storage_mr2, _, pval_sens_P_storage_mr2, pval_sens_PET_storage_mr2, pval_sens_Q_storage_mr2 \
            = sig_SensitivityWithStorage(
            df_tmp["Q"].values, df_tmp["date"].values, df_tmp["P"].values,
            df_tmp["PET"].values, use_delta=True, fit_intercept=True, wateryear=wy_str)
        sens_P_avg_mr1, sens_PET_avg_mr1, R2_avg_mr1, _, _, _, pval_sens_P_avg_mr1, pval_sens_PET_avg_mr1 = sig_SensitivityAveraging(
            df_tmp["Q"].values, df_tmp["date"].values, df_tmp["P"].values,
            df_tmp["PET"].values, use_delta=False, fit_intercept=False, wateryear=wy_str, n=5)
        sens_P_avg_mr2, sens_PET_avg_mr2, R2_avg_mr2, _, _, _, pval_sens_P_avg_mr2, pval_sens_PET_avg_mr2 = sig_SensitivityAveraging(
            df_tmp["Q"].values, df_tmp["date"].values, df_tmp["P"].values,
            df_tmp["PET"].values, use_delta=True, fit_intercept=True, wateryear=wy_str, n=5)

        # calculate sensitivities over time
        df_time_mr1 = sig_SensitivityOverTime(
            df_tmp["Q"].values, df_tmp["date"].values, df_tmp["P"].values,
            df_tmp["PET"].values, id, plot_results=False, use_delta=False, fit_intercept=False, window_years=20)

        df_time_mr2 = sig_SensitivityOverTime(
            df_tmp["Q"].values, df_tmp["date"].values, df_tmp["P"].values,
            df_tmp["PET"].values, id, plot_results=False, use_delta=True, fit_intercept=True, window_years=20)

        # baseflow
        BFI = sig_BFI(df_tmp["Q"].values, df_tmp["date"].values)
        BFI = BFI[0]


    # return a dictionary with all results for this gauge
    return {
        "gauge_id_native": id,
        "gauge_id": gauge_id,
        "perc_complete": perc_complete,
        "neg_count": neg_count,
        "len_years": len_years,
        "frac_snow_control": frac_snow_control,
        "aridity_control": aridity_control,
        "seasonality_index": seasonality_index,
        "P_seasonality_index": P_seasonality_index,
        "mean_P": mean_P,
        "mean_PET": mean_PET,
        "mean_Q": mean_Q,
        "mean_T": mean_T,
        "annual_P": annual_P,
        "annual_PET": annual_PET,
        "annual_Q": annual_Q,
        "annual_T": annual_T,
        "cor_PET_P": cor_PET_P,
        "sensitivity_len": sensitivity_len,
        "sens_P_mr1": sens_P_mr1,
        "sens_PET_mr1": sens_PET_mr1,
        "pval_sens_P_mr1": pval_sens_P_mr1,
        "pval_sens_PET_mr1": pval_sens_PET_mr1,
        "R2_mr1": R2_mr1,
        "sens_P_mr2": sens_P_mr2,
        "sens_PET_mr2": sens_PET_mr2,
        "pval_sens_P_mr2": pval_sens_P_mr2,
        "pval_sens_PET_mr2": pval_sens_PET_mr2,
        "R2_mr2": R2_mr2,
        "sens_P_log": sens_P_log,
        "sens_PET_log": sens_PET_log,
        "pval_sens_P_log": pval_sens_P_log,
        "pval_sens_PET_log": pval_sens_PET_log,
        "R2_log": R2_log,
        "sens_P_storage_mr1": sens_P_storage_mr1,
        "sens_PET_storage_mr1": sens_PET_storage_mr1,
        "sens_Q_storage_mr1": sens_Q_storage_mr1,
        "pval_sens_P_storage_mr1": pval_sens_P_storage_mr1,
        "pval_sens_PET_storage_mr1": pval_sens_PET_storage_mr1,
        "pval_sens_Q_storage_mr1": pval_sens_Q_storage_mr1,
        "R2_storage_mr1": R2_storage_mr1,
        "sens_P_storage_mr2": sens_P_storage_mr2,
        "sens_PET_storage_mr2": sens_PET_storage_mr2,
        "sens_Q_storage_mr2": sens_Q_storage_mr2,
        "pval_sens_P_storage_mr2": pval_sens_P_storage_mr2,
        "pval_sens_PET_storage_mr2": pval_sens_PET_storage_mr2,
        "pval_sens_Q_storage_mr2": pval_sens_Q_storage_mr2,
        "R2_storage_mr2": R2_storage_mr2,
        "sens_P_avg_mr1": sens_P_avg_mr1,
        "sens_PET_avg_mr1": sens_PET_avg_mr1,
        "pval_sens_P_avg_mr1": pval_sens_P_avg_mr1,
        "pval_sens_PET_avg_mr1": pval_sens_PET_avg_mr1,
        "R2_avg_mr1": R2_avg_mr1,
        "sens_P_avg_mr2": sens_P_avg_mr2,
        "sens_PET_avg_mr2": sens_PET_avg_mr2,
        "pval_sens_P_avg_mr2": pval_sens_P_avg_mr2,
        "pval_sens_PET_avg_mr2": pval_sens_PET_avg_mr2,
        "R2_avg_mr2": R2_avg_mr2,
        "start_wateryear": df_time_mr1["start_wateryear"].values,
        "end_wateryear": df_time_mr1["end_wateryear"].values,
        "mean_P_over_time": df_time_mr1["mean_P"].values,
        "mean_PET_over_time": df_time_mr1["mean_PET"].values,
        "mean_Q_over_time": df_time_mr1["mean_Q"].values,
        "aridity_over_time": df_time_mr1["aridity"].values,
        "cor_PET_P_over_time": df_time_mr1["P_PET_correlation"].values,
        "sens_P_over_time_mr1": df_time_mr1["sens_P"].values,
        "sens_PET_over_time_mr1": df_time_mr1["sens_PET"].values,
        "pval_sens_P_over_time_mr1": df_time_mr1["pval_sens_P"].values,
        "pval_sens_PET_over_time_mr1": df_time_mr1["pval_sens_PET"].values,
        "R2_over_time_mr1": df_time_mr1["R2"].values,
        "sens_P_over_time_mr2": df_time_mr2["sens_P"].values,
        "sens_PET_over_time_mr2": df_time_mr2["sens_PET"].values,
        "pval_sens_P_over_time_mr2": df_time_mr2["pval_sens_P"].values,
        "pval_sens_PET_over_time_mr2": df_time_mr2["pval_sens_PET"].values,
        "R2_over_time_mr2": df_time_mr2["R2"].values,
        "BFI": BFI
    }
