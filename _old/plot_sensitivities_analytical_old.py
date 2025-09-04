import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from pylr2 import regress2

# prepare data
data_path = "D:/Data/"

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)


# Define functions for major axis regression
def ma_regression(x, y):
    results = regress2(x, y, _method_type_2="reduced major axis")
    slope = results['slope']
    intercept = results['intercept']
    # residuals = (y - (slope * x + intercept)) / np.sqrt(1 + slope ** 2)
    residuals = (y - (slope * x + intercept))
    return slope, intercept, residuals


def partial_ma_sensitivity(resid_x, resid_y):
    slope, _, _ = ma_regression(resid_x, resid_y)
    return slope


# Define functions to calculate streamflow and sensitivities
def calculate_streamflow(P, PET, n):
    return P - (P ** -n + PET ** -n) ** (-1 / n)


def calculate_sensitivities(P, PET, n):
    dQ_dP = 1 - (1 + (P / PET) ** n) ** ((-1 / n) - 1)
    dQ_dPET = - (1 + (PET / P) ** n) ** ((-1 / n) - 1)
    return dQ_dP, dQ_dPET


# Generate synthetic data based on correlation and noise
def generate_data(p_mean, pet_mean, corr, noise):
    sd_P = 0.1 * p_mean
    sd_PET = 0.1 * pet_mean
    cov_P_PET = corr * sd_P * sd_PET
    sigma = np.array([[sd_P ** 2, cov_P_PET], [cov_P_PET, sd_PET ** 2]])
    mu = np.array([p_mean, pet_mean])
    sample = multivariate_normal(mu, sigma).rvs(50)
    df_art = pd.DataFrame(sample, columns=["P", "PET"])
    df_art["Q"] = calculate_streamflow(df_art["P"].values, df_art["PET"].values, 2)

    # Add noise if specified
    if noise > 0:
        df_art["Q"] *= (1 + np.random.normal(0, noise, len(df_art)))
        df_art["P"] *= (1 + np.random.normal(0, noise, len(df_art)))
        df_art["PET"] *= (1 + np.random.normal(0, noise, len(df_art)))

    # Calculate streamflow

    return df_art


# Function to calculate relative error
def calculate_relative_error(estimated, theoretical):
    return 100 * ((estimated - theoretical) / theoretical)


# Main plotting function
def plot_sensitivities():
    # Define parameter ranges and settings
    p_means = np.linspace(100, 2100, 20)  # P values
    pet_means = np.linspace(100, 2100, 20)  # PET values
    correlations = [-0.5, 0.0, 0.5]  # Correlation values
    noise_levels = [0.0, 0.025]  # 95% will lie within two standard deviations (±2*X%) of the mean.

    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)  # For aridity plot
    fig_P, axs_P = plt.subplots(3, 2, figsize=(6, 8))  # For P sensitivities
    fig_PET, axs_PET = plt.subplots(3, 2, figsize=(6, 8))  # For PET sensitivities

    for j, noise in enumerate(noise_levels):  # Columns: Noise levels
        for i, corr in enumerate(correlations):  # Rows: Different correlations
            results = []

            relative_errors_P = {method: [] for method in [
                "Nonparametric", "Single Reg.", "Mult. Reg. #1",
                "Mult. Reg. #2", "Mult. Reg. #3"
            ]}
            relative_errors_PET = {method: [] for method in [
                "Nonparametric", "Single Reg.", "Mult. Reg. #1",
                "Mult. Reg. #2", "Mult. Reg. #3"
            ]}

            for p_mean in p_means:
                for pet_mean in pet_means:
                    # Generate data
                    df_art = generate_data(p_mean=p_mean, pet_mean=pet_mean, corr=corr, noise=noise)

                    # Calculate theoretical sensitivities
                    sens_P_theory, sens_PET_theory = calculate_sensitivities(p_mean, pet_mean, n=2)

                    # Calculate aridity index (PET/P)
                    aridity_index = pet_mean / p_mean

                    # Method 1: Nonparametric estimation (median of ratios)
                    sens_P_method1 = np.median((df_art["Q"] - df_art["Q"].mean()) / (df_art["P"] - df_art["P"].mean()))
                    sens_PET_method1 = np.median(
                        (df_art["Q"] - df_art["Q"].mean()) / (df_art["PET"] - df_art["PET"].mean()))

                    # Method 2: Single regressions (independent variables separately)
                    model_P_Q = LinearRegression().fit(df_art[['P']], df_art['Q'])
                    model_PET_Q = LinearRegression().fit(df_art[['PET']], df_art['Q'])
                    sens_P_method2 = model_P_Q.coef_[0]
                    sens_PET_method2 = model_PET_Q.coef_[0]

                    # Method 3: Multiple regression (non-centered data)
                    X_noncentered = df_art[["P", "PET"]]
                    y_noncentered = df_art["Q"]
                    sens_P_method3, sens_PET_method3 = LinearRegression(fit_intercept=False).fit(X_noncentered,
                                                                                                 y_noncentered).coef_

                    # Method 4: Multiple regression (centered data)
                    X_centered = X_noncentered - X_noncentered.mean()
                    y_centered = y_noncentered - y_noncentered.mean()
                    sens_P_method4, sens_PET_method4 = LinearRegression(fit_intercept=False).fit(X_centered,
                                                                                                 y_centered).coef_

                    # Method 5: Log-log regression (elasticities converted to sensitivities)
                    log_X = np.log(df_art[["P", "PET"]])
                    log_y = np.log(df_art["Q"])
                    log_model = LinearRegression(fit_intercept=True).fit(log_X, log_y)
                    sens_P_method5 = log_model.coef_[0] * df_art["Q"].mean() / df_art["P"].mean()
                    sens_PET_method5 = log_model.coef_[1] * df_art["Q"].mean() / df_art["PET"].mean()
                    # Calculate residuals for partial regression

                    '''
                    # Method 6: major axis regression
                    _, _, resid_Q_after_PET = ma_regression(df_art['PET'], df_art['Q'])
                    _, _, resid_Q_after_P = ma_regression(df_art['P'], df_art['Q'])
                    _, _, resid_PET_after_P = ma_regression(df_art['P'], df_art['PET'])
                    _, _, resid_P_after_PET = ma_regression(df_art['PET'], df_art['P'])
                    sens_P_method6 = partial_ma_sensitivity(resid_P_after_PET, resid_Q_after_PET)
                    sens_PET_method6 = partial_ma_sensitivity(resid_PET_after_P, resid_Q_after_P)

                    # Method 7: Multi-year averaging
                    n_years = 5
                    n_blocks = len(y_noncentered) // n_years
                    data_avg = pd.DataFrame({
                        "Q": y_noncentered.iloc[:n_blocks * n_years],  # Trim to complete blocks
                        "P": X_noncentered['P'].iloc[:n_blocks * n_years],
                        "PET": X_noncentered['PET'].iloc[:n_blocks * n_years]
                    })
                    data_avg['block'] = np.repeat(np.arange(n_blocks), n_years)
                    averaged_data = data_avg.groupby('block').mean()

                    if not averaged_data.empty:
                        avg_model = LinearRegression(fit_intercept=False).fit(averaged_data[["P", "PET"]],
                                                                             averaged_data["Q"])
                        sens_P_method7 = avg_model.coef_[0]
                        sens_PET_method7 = avg_model.coef_[1]
                    else:
                        sens_P_method7 = np.nan
                        sens_PET_method7 = np.nan
                    '''

                    # Append results for all methods and theoretical values
                    results.extend([
                        {"aridity_index": aridity_index,
                         "method": "Theoretical", "sens_P": sens_P_theory,
                         "sens_PET": sens_PET_theory},
                        {"aridity_index": aridity_index,
                         "method": "Nonparametric", "sens_P": sens_P_method1,
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
                         "method": "Mult. Reg. #3",
                         "sens_P": sens_P_method5,
                         "sens_PET": sens_PET_method5}
                    ])

                    # Calculate relative errors and store them for P and PET sensitivities
                    relative_errors_P["Nonparametric"].append(calculate_relative_error(sens_P_method1, sens_P_theory))
                    relative_errors_P["Single Reg."].append(calculate_relative_error(sens_P_method2, sens_P_theory))
                    relative_errors_P["Mult. Reg. #1"].append(calculate_relative_error(sens_P_method3, sens_P_theory))
                    relative_errors_P["Mult. Reg. #2"].append(calculate_relative_error(sens_P_method4, sens_P_theory))
                    relative_errors_P["Mult. Reg. #3"].append(calculate_relative_error(sens_P_method5, sens_P_theory))

                    relative_errors_PET["Nonparametric"].append(
                        calculate_relative_error(sens_PET_method1, sens_PET_theory))
                    relative_errors_PET["Single Reg."].append(
                        calculate_relative_error(sens_PET_method2, sens_PET_theory))
                    relative_errors_PET["Mult. Reg. #1"].append(
                        calculate_relative_error(sens_PET_method3, sens_PET_theory))
                    relative_errors_PET["Mult. Reg. #2"].append(
                        calculate_relative_error(sens_PET_method4, sens_PET_theory))
                    relative_errors_PET["Mult. Reg. #3"].append(
                        calculate_relative_error(sens_PET_method5, sens_PET_theory))

            # Average relative errors for each method
            avg_relative_errors_P = {method: np.mean(errors) for method, errors in relative_errors_P.items()}
            avg_relative_errors_PET = {method: np.mean(errors) for method, errors in relative_errors_PET.items()}

            # Define custom colors for each method
            custom_palette = {
                'Theoretical': 'black',
                'Nonparametric': 'lightgrey',
                'Single Reg.': 'grey',
                'Mult. Reg. #1': 'tab:orange',
                'Mult. Reg. #2': 'tab:red',
                'Mult. Reg. #3': 'tab:purple',
                'Averaging': 'tab:blue'
            }

            # Convert results to DataFrame for plotting
            results_df = pd.DataFrame(results)
            results_melted = results_df.melt(
                id_vars=["aridity_index", "method"],
                value_vars=["sens_P", "sens_PET"],
                var_name="variable",
                value_name="sensitivity"
            )

            # Plot sensitivities vs aridity index for each method and variable
            ax = axs[i][j]
            sns.scatterplot(
                data=results_melted,
                x="aridity_index",
                y="sensitivity",
                hue="method",
                style="variable",
                ax=ax,
                palette=custom_palette,
                alpha=0.3,
                s=5,
                edgecolor=None,
                zorder=2
            )
            ax.legend().remove()
            ax.set_title(f"Corr={corr}, Noise={noise}")
            ax.set_xlabel("Aridity index [-]")
            ax.set_ylabel("Sensitivity [-]")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlim(0.1, 10)
            ax.set_ylim(-1.5, 1.5)

            # add line through scatter plot
            for (method, variable), group_data in results_melted.groupby(["method", "variable"]):
                color = custom_palette.get(method, "black")  # fallback to black if method is missing
                sorted_idx = np.argsort(group_data["aridity_index"])
                lowess = sm.nonparametric.lowess
                smoothed = lowess(
                    group_data["sensitivity"],
                    group_data["aridity_index"],
                    frac=0.2
                )
                ax.plot(
                    smoothed[:, 0], smoothed[:, 1],
                    color=color,
                    linewidth=2,
                    alpha=0.9,
                    linestyle="solid" if variable == "sens_P" else "dashed",  # optional: different style per variable
                    zorder=5
                )

            # Plot bar plot of average relative errors for P sensitivities
            ax_p = axs_P[i][j]
            sns.barplot(
                x=list(avg_relative_errors_P.keys()),
                y=list(avg_relative_errors_P.values()),
                ax=ax_p,
                hue=list(avg_relative_errors_P.keys()),
                palette=custom_palette  # "viridis"
            )
            ax_p.set_title(f"P Sensitivities\nCorr={corr}, Noise={noise}")
            ax_p.set_ylabel("Relative Error [%]")
            ax_p.tick_params(axis='x', rotation=45)
            ax_p.set_ylim(-20, 20)

            # Plot bar plot of average relative errors for PET sensitivities
            ax_pet = axs_PET[i][j]
            sns.barplot(
                x=list(avg_relative_errors_PET.keys()),
                y=list(avg_relative_errors_PET.values()),
                ax=ax_pet,
                hue=list(avg_relative_errors_P.keys()),
                palette=custom_palette  # "viridis"
            )
            ax_pet.set_title(f"PET Sensitivities\nCorr={corr}, Noise={noise}")
            ax_pet.set_ylabel("Relative Error [%]")
            ax_pet.tick_params(axis='x', rotation=45)
            ax_pet.set_ylim(-20, 20)

    ## Create a single legend outside all subplots
    handles, labels = axs[0][0].get_legend_handles_labels()  # Get handles and labels from one of the subplots
    fig.legend(handles, labels, loc="center right", title="Methods", bbox_to_anchor=(1.15, 0.5))
    # plt.tight_layout(rect=[0, 0, 0.85, 1])

    fig.tight_layout()
    fig_P.tight_layout()
    fig_PET.tight_layout()

    fig.savefig(figures_path + "theoretical_sensitivities_aridity_test.png", dpi=600, bbox_inches='tight')
    fig_P.savefig(figures_path + "theoretical_sensitivities_bar_P_test.png", dpi=600, bbox_inches='tight')
    fig_PET.savefig(figures_path + "theoretical_sensitivities_bar_PET_test.png", dpi=600, bbox_inches='tight')


# Call the plotting function to display the figure
plot_sensitivities()
