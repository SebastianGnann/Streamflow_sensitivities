import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from functions import util_Turc
from scipy.stats import multivariate_normal
import seaborn as sns
import statsmodels.api as sm
from matplotlib.lines import Line2D
from functions.helper_ODR import run_odr  # Your ODR function as defined previously

# Note: make sure the run_odr function is correctly implemented and imported.

# generate synthetic data based on correlation and noise
def generate_data(p_mean, pet_mean, corr, noise, n, years):
    sd_P = 0.1 * p_mean
    sd_PET = 0.1 * pet_mean
    cov_P_PET = corr * sd_P * sd_PET
    sigma = np.array([[sd_P ** 2, cov_P_PET], [cov_P_PET, sd_PET ** 2]])
    mu = np.array([p_mean, pet_mean])
    sample = multivariate_normal(mu, sigma).rvs(years)
    df_art = pd.DataFrame(sample, columns=["P", "PET"])
    df_art["Q"] = util_Turc.calculate_streamflow(df_art["P"].values, df_art["PET"].values, n)
    if noise > 0:
        df_art["Q"] *= (1 + np.random.normal(0, noise, len(df_art)))
        df_art["P"] *= (1 + np.random.normal(0, noise, len(df_art)))
        df_art["PET"] *= (1 + np.random.normal(0, noise, len(df_art)))
    return df_art

def calculate_relative_error(estimated, theoretical):
    return 100 * ((estimated - theoretical) / theoretical)

def plot_sensitivities():
    p_means = np.linspace(100, 2100, 10)
    pet_means = np.linspace(100, 2100, 10)
    correlations = [-0.5, 0.0, 0.5]
    noise_levels = [0.0, 0.025]
    n = 2.5

    method_order = [
        "Analytical",
        "Nonparam.",
        "Single Reg.",
        "Mult. Reg. #1",
        "Mult. Reg. #2",
        "Mult. Reg. Log",
        "ODR No Intercept",
        "ODR With Intercept"
    ]

    custom_palette = {
        'Analytical': 'black',
        'Nonparam.': 'gainsboro',
        'Single Reg.': 'silver',
        'Mult. Reg. #1': '#f6705b',
        'Mult. Reg. #2': '#b63679',
        'Mult. Reg. Log': '#63197f',
        'ODR No Intercept': '#1f77b4',
        'ODR With Intercept': '#ff7f0e'
    }

    avg_errors_summary = []
    avg_pvalues_summary = []
    avg_r2_summary = []

    fig_P, axs_P = plt.subplots(3, 2, figsize=(8, 7), sharex=True, sharey=True)
    method_handles_P = {}
    fig_PET, axs_PET = plt.subplots(3, 2, figsize=(8, 7), sharex=True, sharey=True)
    method_handles_PET = {}

    fig_bar_P, axs_bar_P = plt.subplots(3, 2, figsize=(5, 8), sharex=True, sharey=True)
    fig_bar_PET, axs_bar_PET = plt.subplots(3, 2, figsize=(5, 8), sharex=True, sharey=True)

    # New: Bar plots for p-values and R2
    fig_bar_pvalues, axs_bar_pvalues = plt.subplots(3, 2, figsize=(5, 8), sharex=True, sharey=True)
    fig_bar_r2, axs_bar_r2 = plt.subplots(3, 2, figsize=(5, 8), sharex=True, sharey=True)

    for j, noise in enumerate(noise_levels):
        for i, corr in enumerate(correlations):
            results = []
            methods = [
                "Nonparam.",
                "Single Reg.",
                "Mult. Reg. #1",
                "Mult. Reg. #2",
                "Mult. Reg. Log",
                "ODR No Intercept",
                "ODR With Intercept"
            ]
            relative_errors_P = {m: [] for m in methods}
            relative_errors_PET = {m: [] for m in methods}

            # New containers for p-values and R2 (only for multivariate)
            pvalues_P = {m: [] for m in methods if "Mult" in m or "ODR" in m}
            pvalues_PET = {m: [] for m in methods if "Mult" in m or "ODR" in m}
            r2_values = {m: [] for m in methods if "Mult" in m or "ODR" in m}

            for p_mean in p_means:
                for pet_mean in pet_means:
                    df_art = generate_data(p_mean=p_mean, pet_mean=pet_mean, corr=corr, noise=noise, n=n, years=50)
                    sens_P_theory, sens_PET_theory = util_Turc.calculate_sensitivities(p_mean, pet_mean, n)
                    aridity_index = pet_mean / p_mean

                    sens_P_method1 = np.median((df_art["Q"] - df_art["Q"].mean()) / (df_art["P"] - df_art["P"].mean()))
                    sens_PET_method1 = np.median(
                        (df_art["Q"] - df_art["Q"].mean()) / (df_art["PET"] - df_art["PET"].mean()))

                    # Method 2: Simple regression P
                    X_P = (df_art[["P"]] - df_art["P"].mean())
                    y = (df_art[["Q"]] - df_art["Q"].mean())
                    model_P_Q = sm.OLS(y.values.ravel(), X_P.values).fit()
                    sens_P_method2 = model_P_Q.params[0]

                    # Method 2: Simple regression PET
                    X_PET = (df_art[["PET"]] - df_art["PET"].mean())
                    model_PET_Q = sm.OLS(y.values.ravel(), X_PET.values).fit()
                    sens_PET_method2 = model_PET_Q.params[0]

                    # Method 3: Multivariate regression non-centered
                    X_noncentered = df_art[["P", "PET"]].values
                    y_noncentered = df_art["Q"].values
                    model_multi = sm.OLS(y_noncentered, X_noncentered).fit()
                    sens_P_method3 = model_multi.params[0]
                    sens_PET_method3 = model_multi.params[1]
                    p_value_P_method3 = model_multi.pvalues[0]
                    p_value_PET_method3 = model_multi.pvalues[1]
                    r2_P_method3 = model_multi.rsquared  # single r2 for multivariate

                    # Method 4: Multivariate regression centered
                    X_centered = (df_art[["P", "PET"]] - df_art[["P", "PET"]].mean()).values
                    y_centered = (df_art["Q"] - df_art["Q"].mean()).values
                    model_multi_centered = sm.OLS(y_centered, X_centered).fit()
                    sens_P_method4 = model_multi_centered.params[0]
                    sens_PET_method4 = model_multi_centered.params[1]
                    p_value_P_method4 = model_multi_centered.pvalues[0]
                    p_value_PET_method4 = model_multi_centered.pvalues[1]
                    r2_P_method4 = model_multi_centered.rsquared

                    # Method 5: Log-log regression with intercept
                    log_X = np.log(df_art[["P", "PET"]].values)
                    log_y = np.log(df_art["Q"].values)
                    log_X_with_const = sm.add_constant(log_X)
                    log_model = sm.OLS(log_y, log_X_with_const).fit()
                    sens_P_method5 = log_model.params[1] * df_art["Q"].mean() / df_art["P"].mean()
                    sens_PET_method5 = log_model.params[2] * df_art["Q"].mean() / df_art["PET"].mean()
                    p_value_P_method5 = log_model.pvalues[1]
                    p_value_PET_method5 = log_model.pvalues[2]
                    r2_P_method5 = log_model.rsquared

                    # Method 6: ODR no intercept
                    sens_P_method6, sens_PET_method6, p_values_method6, r2_method6 = run_odr(df_art,
                                                                                             method="without_intercept")
                    p_value_P_method6, p_value_PET_method6 = p_values_method6

                    # Method 7: ODR with intercept
                    sens_P_method7, sens_PET_method7, intercept_method7, p_values_method7, r2_method7 = run_odr(df_art,
                                                                                                                method="with_intercept")
                    p_value_P_method7, p_value_PET_method7, p_value_intercept_method7 = p_values_method7

                    results.extend([
                        {"aridity_index": aridity_index, "method": "Analytical", "sens_P": sens_P_theory,
                         "sens_PET": sens_PET_theory},
                        {"aridity_index": aridity_index, "method": "Nonparam.", "sens_P": sens_P_method1,
                         "sens_PET": sens_PET_method1},
                        {"aridity_index": aridity_index, "method": "Single Reg.", "sens_P": sens_P_method2,
                         "sens_PET": sens_PET_method2},
                        {"aridity_index": aridity_index, "method": "Mult. Reg. #1", "sens_P": sens_P_method3,
                         "sens_PET": sens_PET_method3},
                        {"aridity_index": aridity_index, "method": "Mult. Reg. #2", "sens_P": sens_P_method4,
                         "sens_PET": sens_PET_method4},
                        {"aridity_index": aridity_index, "method": "Mult. Reg. Log", "sens_P": sens_P_method5,
                         "sens_PET": sens_PET_method5},
                        {"aridity_index": aridity_index, "method": "ODR No Intercept", "sens_P": sens_P_method6,
                         "sens_PET": sens_PET_method6},
                        {"aridity_index": aridity_index, "method": "ODR With Intercept", "sens_P": sens_P_method7,
                         "sens_PET": sens_PET_method7}
                    ])

                    for method, est, theory in zip(
                            methods,
                            [sens_P_method1, sens_P_method2, sens_P_method3, sens_P_method4, sens_P_method5,
                             sens_P_method6, sens_P_method7],
                            [sens_P_theory] * 7):
                        relative_errors_P[method].append(calculate_relative_error(est, theory))
                    for method, est, theory in zip(
                            methods,
                            [sens_PET_method1, sens_PET_method2, sens_PET_method3, sens_PET_method4, sens_PET_method5,
                             sens_PET_method6, sens_PET_method7],
                            [sens_PET_theory] * 7):
                        relative_errors_PET[method].append(calculate_relative_error(est, theory))

                    # Store p-values and R2 for multivariate + ODR methods only
                    for method, pP, pPET, r2 in zip(
                            ["Mult. Reg. #1", "Mult. Reg. #2", "Mult. Reg. Log", "ODR No Intercept",
                             "ODR With Intercept"],
                            [p_value_P_method3, p_value_P_method4, p_value_P_method5, p_value_P_method6,
                             p_value_P_method7],
                            [p_value_PET_method3, p_value_PET_method4, p_value_PET_method5, p_value_PET_method6,
                             p_value_PET_method7],
                            [r2_P_method3, r2_P_method4, r2_P_method5, r2_method6, r2_method7]
                    ):
                        pvalues_P[method].append(pP)
                        pvalues_PET[method].append(pPET)
                        r2_values[method].append(r2)

            avg_relative_errors_P = {m: np.mean(v) for m, v in relative_errors_P.items()}
            avg_relative_errors_PET = {m: np.mean(v) for m, v in relative_errors_PET.items()}
            avg_pvalues_P = {m: np.mean(v) for m, v in pvalues_P.items()}
            avg_pvalues_PET = {m: np.mean(v) for m, v in pvalues_PET.items()}
            avg_r2 = {m: np.mean(v) for m, v in r2_values.items()}

            # Store averages for summary tables (relative error)
            for method in methods:
                avg_errors_summary.append({
                    "Panel": f"Corr={corr}, Noise={noise}",
                    "Sensitivity": "P",
                    "Method": method,
                    "Avg Relative Error (%)": avg_relative_errors_P[method]
                })
                avg_errors_summary.append({
                    "Panel": f"Corr={corr}, Noise={noise}",
                    "Sensitivity": "PET",
                    "Method": method,
                    "Avg Relative Error (%)": avg_relative_errors_PET[method]
                })

            # Store averages for p-values (only multivariate + ODR)
            for method in avg_pvalues_P.keys():
                avg_pvalues_summary.append({
                    "Panel": f"Corr={corr}, Noise={noise}",
                    "Sensitivity": "P",
                    "Method": method,
                    "Avg p-value": avg_pvalues_P[method]
                })
                avg_pvalues_summary.append({
                    "Panel": f"Corr={corr}, Noise={noise}",
                    "Sensitivity": "PET",
                    "Method": method,
                    "Avg p-value": avg_pvalues_PET[method]
                })

            # Store averages for R2 values (only multivariate + ODR)
            for method in avg_r2.keys():
                avg_r2_summary.append({
                    "Panel": f"Corr={corr}, Noise={noise}",
                    "Method": method,
                    "Avg R2": avg_r2[method]
                })

            results_df = pd.DataFrame(results)

            # P sensitivities panel
            ax_P = axs_P[i, j]
            results_P = results_df.melt(id_vars=["aridity_index", "method"], value_vars=["sens_P"],
                                        var_name="variable", value_name="sensitivity")
            results_P["method"] = pd.Categorical(results_P["method"], categories=method_order, ordered=True)
            sns.scatterplot(data=results_P, x="aridity_index", y="sensitivity", hue="method", hue_order=method_order,
                            ax=ax_P, palette=custom_palette, alpha=0.25, s=2.5, edgecolor=None, zorder=2, legend=False)
            ax_P.set_title(f"Corr={corr}, Noise={noise}", fontsize=10)
            ax_P.set_xlabel(r"$PET$/$P$ [-]")
            ax_P.set_ylabel(r"$s_P$ [-]")
            ax_P.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax_P.set_xscale("log")
            ax_P.set_xlim(0.1, 10)
            ax_P.set_ylim(-0.1, 1.5)
            for method in method_order:
                group_data = results_P[results_P["method"] == method]
                if not group_data.empty:
                    color = custom_palette.get(method, "black")
                    smoothed = sm.nonparametric.lowess(group_data["sensitivity"], group_data["aridity_index"], frac=0.2)
                    line, = ax_P.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2, alpha=0.9,
                                      linestyle="solid", zorder=5, label=method)
                    if method not in method_handles_P:
                        method_handles_P[method] = Line2D([0], [0], color=color, lw=2, linestyle="solid")

            # PET sensitivities panel
            ax_PET = axs_PET[i, j]
            results_PET = results_df.melt(id_vars=["aridity_index", "method"], value_vars=["sens_PET"],
                                          var_name="variable", value_name="sensitivity")
            sns.scatterplot(data=results_PET, x="aridity_index", y="sensitivity", hue="method", hue_order=method_order,
                            ax=ax_PET, palette=custom_palette, alpha=0.25, s=2.5, edgecolor=None, zorder=2,
                            legend=False)
            ax_PET.set_title(f"Corr={corr}, Noise={noise}", fontsize=10)
            ax_PET.set_xlabel(r"$PET$/$P$ [-]")
            ax_PET.set_ylabel(r"$s_{PET}$ [-]")
            ax_PET.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax_PET.set_xscale("log")
            ax_PET.set_xlim(0.1, 10)
            ax_PET.set_ylim(-1.5, 0.1)
            for method in method_order:
                group_data = results_PET[results_PET["method"] == method]
                if not group_data.empty:
                    color = custom_palette.get(method, "black")
                    smoothed = sm.nonparametric.lowess(group_data["sensitivity"], group_data["aridity_index"], frac=0.2)
                    line, = ax_PET.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2, alpha=0.9,
                                        linestyle="solid", zorder=5, label=method)
                    if method not in method_handles_PET:
                        method_handles_PET[method] = Line2D([0], [0], color=color, lw=2, linestyle="solid")

            # Barplots for relative errors P
            ax_bar_P = axs_bar_P[i, j]
            sns.barplot(x=list(avg_relative_errors_P.keys()), y=list(avg_relative_errors_P.values()),
                        ax=ax_bar_P, palette=custom_palette, hue=list(avg_relative_errors_P.keys()))
            ax_bar_P.set_title(f"Corr={corr}, Noise={noise}", fontsize=10)
            ax_bar_P.set_ylabel("Relative Error [%]")
            ax_bar_P.tick_params(axis='x', rotation=45)
            ax_bar_P.set_ylim(-25, 25)

            # Barplots for relative errors PET
            ax_bar_PET = axs_bar_PET[i, j]
            sns.barplot(x=list(avg_relative_errors_PET.keys()), y=list(avg_relative_errors_PET.values()),
                        ax=ax_bar_PET, palette=custom_palette, hue=list(avg_relative_errors_PET.keys()))
            ax_bar_PET.set_title(f"Corr={corr}, Noise={noise}", fontsize=10)
            ax_bar_PET.set_ylabel("Relative Error [%]")
            ax_bar_PET.tick_params(axis='x', rotation=45)
            ax_bar_PET.set_ylim(-25, 25)

            # Barplots for p-values P (multivariate + ODR only)
            ax_bar_pval_P = axs_bar_pvalues[i, j]
            sns.barplot(x=list(avg_pvalues_P.keys()), y=list(avg_pvalues_P.values()),
                        ax=ax_bar_pval_P, palette=custom_palette)
            ax_bar_pval_P.set_title(f"Corr={corr}, Noise={noise} (P p-values)")
            ax_bar_pval_P.tick_params(axis='x', rotation=45)
            ax_bar_pval_P.set_ylim(0, 1)

            # Barplots for p-values PET (multivariate + ODR only)
            ax_bar_pval_PET = axs_bar_pvalues[i, j]
            sns.barplot(x=list(avg_pvalues_PET.keys()), y=list(avg_pvalues_PET.values()),
                        ax=ax_bar_pval_PET, palette=custom_palette)
            ax_bar_pval_PET.set_title(f"Corr={corr}, Noise={noise} (PET p-values)")
            ax_bar_pval_PET.tick_params(axis='x', rotation=45)
            ax_bar_pval_PET.set_ylim(0, 1)

            # Barplots for R2 (multivariate + ODR only)
            ax_bar_r2_ = axs_bar_r2[i, j]
            sns.barplot(x=list(avg_r2.keys()), y=list(avg_r2.values()),
                        ax=ax_bar_r2_, palette=custom_palette)
            ax_bar_r2_.set_title(f"Corr={corr}, Noise={noise} (R2)")
            ax_bar_r2_.tick_params(axis='x', rotation=45)
            ax_bar_r2_.set_ylim(0, 1)

    # Print and save summary CSVs for p-values and R2
    df_avg_pvalues = pd.DataFrame(avg_pvalues_summary)
    df_avg_r2 = pd.DataFrame(avg_r2_summary)

    df_avg_pvalues_pivot = df_avg_pvalues.pivot_table(index=["Panel", "Sensitivity"], columns="Method",
                                                      values="Avg p-value").round(4)
    df_avg_r2_pivot = df_avg_r2.pivot_table(index=["Panel", "Method"], values="Avg R2").round(4)

    pd.set_option('display.max_rows', None)
    print("\nAverage p-values for multivariate methods (P and PET):")
    print(df_avg_pvalues_pivot)
    print("\nAverage R2 for multivariate methods:")
    print(df_avg_r2_pivot)

    df_avg_pvalues_pivot.reset_index().to_csv("results/avg_pvalues_sensitivities.csv", index=False)
    df_avg_r2_pivot.reset_index().to_csv("results/avg_r2_sensitivities.csv", index=False)

    # Legends for sensitivities
    handles_P = [method_handles_P[m] for m in custom_palette.keys() if m in method_handles_P]
    handles_PET = [method_handles_PET[m] for m in custom_palette.keys() if m in method_handles_PET]
    labels = list(custom_palette.keys())

    fig_P.legend(handles_P, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=int(len(labels) / 2))
    fig_PET.legend(handles_PET, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=int(len(labels) / 2))

    fig_bar_P.tight_layout()
    fig_bar_PET.tight_layout()
    fig_bar_pvalues.tight_layout()
    fig_bar_r2.tight_layout()

    figures_path = "../figures/"
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)

    fig_P.savefig(figures_path + "theoretical_sensitivities_aridity_P_full.png", dpi=600, bbox_inches='tight')
    fig_PET.savefig(figures_path + "theoretical_sensitivities_aridity_PET_full.png", dpi=600, bbox_inches='tight')
    fig_bar_P.savefig(figures_path + "theoretical_sensitivities_bar_P_full.png", dpi=600, bbox_inches='tight')
    fig_bar_PET.savefig(figures_path + "theoretical_sensitivities_bar_PET_full.png", dpi=600, bbox_inches='tight')
    fig_bar_pvalues.savefig(figures_path + "theoretical_sensitivities_bar_pvalues_full.png", dpi=600, bbox_inches='tight')
    fig_bar_r2.savefig(figures_path + "theoretical_sensitivities_bar_r2_full.png", dpi=600, bbox_inches='tight')

# Run plotting and save Turc curves plot
util_Turc.plot_Turc_curves()
fig = plt.gcf()
fig.savefig("figures/Turc_curves.png", dpi=600, bbox_inches='tight')

plot_sensitivities()
