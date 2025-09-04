import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from functions import util_Turc
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression, Ridge
import seaborn as sns
import statsmodels.api as sm
from matplotlib.lines import Line2D
#mpl.use('TkAgg')

# prepare data
data_path = "D:/Data/"

# check if folders exist
results_path = "../results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "../figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

# define helper and plotting functions

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
    if noise > 0: # add noise if specified
        df_art["Q"] *= (1 + np.random.normal(0, noise, len(df_art)))
        df_art["P"] *= (1 + np.random.normal(0, noise, len(df_art)))
        df_art["PET"] *= (1 + np.random.normal(0, noise, len(df_art)))
    return df_art

# calculate relative error
def calculate_relative_error(estimated, theoretical):
    return 100 * ((estimated - theoretical) / theoretical)

def plot_sensitivities():
    p_means = np.linspace(100, 2100, 100)
    pet_means = np.linspace(100, 2100, 100)
    correlations = [-0.5, 0.0, 0.5]
    noise_levels = [0.0, 0.025]
    n = 2.5

    method_order = [
        "Analytical",
        "Nonparam.",
        "Single Reg.",
        "Mult. Reg. #1",
        "Mult. Reg. #2",
        "Mult. Reg. Log"
    ]

    custom_palette = {
        'Analytical': 'black',
        'Nonparam.': 'gainsboro',
        'Single Reg.': 'silver',
        'Mult. Reg. #1': '#f6705b',
        'Mult. Reg. #2': '#b63679',
        'Mult. Reg. Log': '#63197f',
    }

    # Store all panel-wise average errors in list to create a summary table later
    avg_errors_summary = []

    # --- SENSITIVITY PANELS ---
    fig_P, axs_P = plt.subplots(3, 2, figsize=(8, 7), sharex=True, sharey=True)
    method_handles_P = {}
    fig_PET, axs_PET = plt.subplots(3, 2, figsize=(8, 7), sharex=True, sharey=True)
    method_handles_PET = {}

    # --- BAR PLOTS ---
    fig_bar_P, axs_bar_P = plt.subplots(3, 2, figsize=(5, 8), sharex=True, sharey=True)
    fig_bar_PET, axs_bar_PET = plt.subplots(3, 2, figsize=(5, 8), sharex=True, sharey=True)

    for j, noise in enumerate(noise_levels):
        for i, corr in enumerate(correlations):
            results = []
            # Only store errors for non-theoretical methods
            methods = ["Nonparam.", "Single Reg.", "Mult. Reg. #1", "Mult. Reg. #2", "Mult. Reg. Log"]
            relative_errors_P = {method: [] for method in methods}
            relative_errors_PET = {method: [] for method in methods}

            for p_mean in p_means:
                for pet_mean in pet_means:
                    df_art = generate_data(p_mean=p_mean, pet_mean=pet_mean, corr=corr, noise=noise, n=n, years=50)
                    sens_P_theory, sens_PET_theory = util_Turc.calculate_sensitivities(p_mean, pet_mean, n)
                    aridity_index = pet_mean / p_mean

                    sens_P_method1 = np.median((df_art["Q"] - df_art["Q"].mean()) / (df_art["P"] - df_art["P"].mean()))
                    sens_PET_method1 = np.median((df_art["Q"] - df_art["Q"].mean()) / (df_art["PET"] - df_art["PET"].mean()))

                    model_P_Q = LinearRegression(fit_intercept=False).fit(df_art[["P"]]-df_art["P"].mean(), df_art[["Q"]]-df_art["Q"].mean())
                    model_PET_Q = LinearRegression(fit_intercept=False).fit(df_art[["PET"]]-df_art["PET"].mean(), df_art[["Q"]]-df_art["Q"].mean())
                    sens_P_method2 = model_P_Q.coef_[0]
                    sens_PET_method2 = model_PET_Q.coef_[0]

                    X_noncentered = df_art[["P", "PET"]]
                    y_noncentered = df_art["Q"]
                    sens_P_method3, sens_PET_method3 = LinearRegression(fit_intercept=False).fit(X_noncentered, y_noncentered).coef_

                    X_centered = X_noncentered - X_noncentered.mean()
                    y_centered = y_noncentered - y_noncentered.mean()
                    sens_P_method4, sens_PET_method4 = LinearRegression(fit_intercept=False).fit(X_centered, y_centered).coef_

                    log_X = np.log(df_art[["P", "PET"]])
                    log_y = np.log(df_art["Q"])
                    log_model = LinearRegression(fit_intercept=True).fit(log_X, log_y)
                    sens_P_method5 = log_model.coef_[0] * df_art["Q"].mean() / df_art["P"].mean()
                    sens_PET_method5 = log_model.coef_[1] * df_art["Q"].mean() / df_art["PET"].mean()

                    results.extend([
                        {"aridity_index": aridity_index,
                         "method": "Analytical", "sens_P": sens_P_theory,
                         "sens_PET": sens_PET_theory},
                        {"aridity_index": aridity_index,
                         "method": "Nonparam.", "sens_P": sens_P_method1,
                         "sens_PET": sens_PET_method1},
                        {"aridity_index": aridity_index,
                         "method": "Single Reg.", "sens_P": sens_P_method2,
                         "sens_PET": sens_PET_method2},
                        {"aridity_index": aridity_index,
                         "method": "Mult. Reg. #1",
                         "sens_P": sens_P_method3,
                         "sens_PET": sens_PET_method3},
                        {"aridity_index": aridity_index,
                         "method": "Mult. Reg. #2",
                         "sens_P": sens_P_method4,
                         "sens_PET": sens_PET_method4},
                        {"aridity_index": aridity_index,
                         "method": "Mult. Reg. Log",
                         "sens_P": sens_P_method5,
                         "sens_PET": sens_PET_method5}
                    ])

                    # Store relative errors for non-theoretical methods
                    for method, est, theory in zip(
                        methods,
                        [sens_P_method1, sens_P_method2, sens_P_method3, sens_P_method4, sens_P_method5],
                        [sens_P_theory]*5
                    ):
                        relative_errors_P[method].append(calculate_relative_error(est, theory))
                    for method, est, theory in zip(
                        methods,
                        [sens_PET_method1, sens_PET_method2, sens_PET_method3, sens_PET_method4, sens_PET_method5],
                        [sens_PET_theory]*5
                    ):
                        relative_errors_PET[method].append(calculate_relative_error(est, theory))

            avg_relative_errors_P = {method: np.mean(errors) for method, errors in relative_errors_P.items()}
            avg_relative_errors_PET = {method: np.mean(errors) for method, errors in relative_errors_PET.items()}

            # --- Store averages for this panel for summary table ---
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

            results_df = pd.DataFrame(results)

            # --- P sensitivities panel ---
            ax_P = axs_P[i, j]
            results_P = results_df.melt(id_vars=["aridity_index", "method"], value_vars=["sens_P"], var_name="variable", value_name="sensitivity")
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
                if group_data.empty:
                    continue
                color = custom_palette.get(method, "black")
                smoothed = sm.nonparametric.lowess(group_data["sensitivity"],group_data["aridity_index"],frac=0.2)
                [line] = ax_P.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2, alpha=0.9,
                                   linestyle="solid", zorder=5, label=method)
                if method not in method_handles_P:
                    method_handles_P[method] = Line2D([0], [0], color=color, lw=2, linestyle="solid")

            # --- PET sensitivities panel ---
            ax_PET = axs_PET[i, j]
            results_PET = results_df.melt(id_vars=["aridity_index", "method"], value_vars=["sens_PET"], var_name="variable", value_name="sensitivity")
            sns.scatterplot(data=results_PET, x="aridity_index", y="sensitivity", hue="method", hue_order=method_order,
                            ax=ax_PET, palette=custom_palette, alpha=0.25, s=2.5, edgecolor=None, zorder=2, legend=False)
            ax_PET.set_title(f"Corr={corr}, Noise={noise}", fontsize=10)
            ax_PET.set_xlabel(r"$PET$/$P$ [-]")
            ax_PET.set_ylabel(r"$s_{PET}$ [-]")
            ax_PET.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax_PET.set_xscale("log")
            ax_PET.set_xlim(0.1, 10)
            ax_PET.set_ylim(-1.5, 0.1)

            for method in method_order:
                group_data = results_PET[results_PET["method"] == method]
                if group_data.empty:
                    continue
                color = custom_palette.get(method, "black")
                smoothed = sm.nonparametric.lowess(group_data["sensitivity"], group_data["aridity_index"], frac=0.2)
                [line] = ax_PET.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2, alpha=0.9,
                                     linestyle="solid", zorder=5, label=method)
                if method not in method_handles_PET:
                    method_handles_PET[method] = Line2D([0], [0], color=color, lw=2, linestyle="solid")

            # --- Barplots for P sensitivities ---
            ax_bar_P = axs_bar_P[i, j]
            sns.barplot(x=list(avg_relative_errors_P.keys()), y=list(avg_relative_errors_P.values()),
                        ax=ax_bar_P, palette=custom_palette, hue=list(avg_relative_errors_P.keys()),)
            ax_bar_P.set_title(f"Corr={corr}, Noise={noise}", fontsize=10)
            ax_bar_P.set_ylabel("Relative Error [%]")
            ax_bar_P.tick_params(axis='x', rotation=45)
            ax_bar_P.set_ylim(-25, 25)

            # --- Barplots for PET sensitivities ---
            ax_bar_PET = axs_bar_PET[i, j]
            sns.barplot(x=list(avg_relative_errors_PET.keys()), y=list(avg_relative_errors_PET.values()),
                        ax=ax_bar_PET, palette=custom_palette, hue=list(avg_relative_errors_PET.keys()),)
            ax_bar_PET.set_title(f"Corr={corr}, Noise={noise}", fontsize=10)
            ax_bar_PET.set_ylabel("Relative Error [%]")
            ax_bar_PET.tick_params(axis='x', rotation=45)
            ax_bar_PET.set_ylim(-25, 25)

    # --- After all panels: summary table and CSV ---
    df_avg_errors = pd.DataFrame(avg_errors_summary)
    df_avg_errors_pivot = df_avg_errors.pivot_table(
        index=["Panel", "Sensitivity"], columns="Method", values="Avg Relative Error (%)"    ).round(2)
    pd.set_option('display.max_rows', None)
    print("\nAverage relative errors for each panel (category):")
    print(df_avg_errors_pivot)
    df_avg_errors_pivot.reset_index().to_csv("results/avg_relative_errors_sensitivities.csv", index=False)

    # --- Legends using LOESS lines only ---
    handles_P = [method_handles_P[m] for m in custom_palette.keys() if m in method_handles_P]
    handles_PET = [method_handles_PET[m] for m in custom_palette.keys() if m in method_handles_PET]
    labels = list(custom_palette.keys())
    fig_P.legend(handles_P, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(labels)/2)
    #fig_P.tight_layout(rect=[0, 0.1, 1, 1])
    fig_PET.legend(handles_PET, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(labels)/2)
    #fig_PET.tight_layout(rect=[0, 0.07, 1, 1])
    fig_bar_P.tight_layout()
    fig_bar_PET.tight_layout()

    # Save plots
    figures_path = "../figures/"
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    fig_P.savefig(figures_path + "theoretical_sensitivities_aridity_P.png", dpi=600, bbox_inches='tight')
    fig_PET.savefig(figures_path + "theoretical_sensitivities_aridity_PET.png", dpi=600, bbox_inches='tight')
    fig_bar_P.savefig(figures_path + "theoretical_sensitivities_bar_P.png", dpi=600, bbox_inches='tight')
    fig_bar_PET.savefig(figures_path + "theoretical_sensitivities_bar_PET.png", dpi=600, bbox_inches='tight')

util_Turc.plot_Turc_curves()
fig = plt.gcf()
fig.savefig(figures_path + "Turc_curves.png", dpi=600, bbox_inches='tight')

plot_sensitivities()
