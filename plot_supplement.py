import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from functions import util_Turc
import re
import functions.plotting_fcts as pf
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
def fmt_func(x, pos):
    return f"{round(x, 1):.1f}"
#mpl.use('TkAgg')

# plots most of the figures in the supplement
# some supplementary plots were also made using the other scripts, e.g. if the same plots were made with a different regression method

# prepare data
data_path = "D:/Data/"

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

# load data
df_CAMELS_US = pd.read_csv(results_path + 'CAMELS_US_sensitivities.csv')
df_CAMELS_US["country"] = 2
# take all catchments

df_CAMELS_GB = pd.read_csv(results_path + 'CAMELS_GB_sensitivities.csv')
df_CAMELS_GB["country"] = 3
# UKBN
df_UKBN = pd.read_csv("./results/UKBN_Station_List_vUKBN2.0_1.csv")
df_CAMELS_GB = df_CAMELS_GB[df_CAMELS_GB["gauge_id_native"].isin(df_UKBN["Station"].values)]

df_CAMELS_DE = pd.read_csv(results_path + 'CAMELS_DE_sensitivities.csv')
df_CAMELS_DE["country"] = 4
# ROBIN catchments
df_ROBIN = pd.read_csv("D:/Python/ROBIN_CAMELS_DE/results/camels_de_ROBIN.csv")
df_CAMELS_DE = df_CAMELS_DE[df_CAMELS_DE["gauge_id_native"].isin(df_ROBIN["ID"].values)]

df_CAMELS_AUS = pd.read_csv(results_path + 'CAMELS_AUS_sensitivities.csv')
df_CAMELS_AUS["country"] = 5
df_humans = pd.read_csv("D:/Data/CAMELS_AUS_v2/04_attributes/04_attributes/CatchmentAttributes_04_AnthropogenicInfluences.csv", sep=',', skiprows=0, encoding='latin-1')
df_humans.rename(columns={'station_id': 'gauge_id_native'}, inplace=True)
df_CAMELS_AUS = pd.merge(df_CAMELS_AUS, df_humans, on='gauge_id_native')
df_CAMELS_AUS = df_CAMELS_AUS[df_CAMELS_AUS["river_di"]<0.2]

df = pd.concat([df_CAMELS_US, df_CAMELS_GB, df_CAMELS_DE, df_CAMELS_AUS], ignore_index=True)

# filter catchments
df = df[df["perc_complete"] > 0.95]
df = df[df["len_years"] > 30]
df = df[df["frac_snow_control"] < 0.2]
df = df[df["mean_P"] > df["mean_Q"]]
# remove catch
df = df.reset_index()

n = 2.2 # Turc-Pike parameter

# transform data
df["annual_P"] = df["annual_P"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["annual_PET"] = df["annual_PET"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["annual_Q"] = df["annual_Q"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["annual_T"] = df["annual_T"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["start_wateryear"] = df["start_wateryear"].apply(lambda x: np.array([float(y) for y in re.findall(r"\d{4}", x)]))
df["end_wateryear"] = df["end_wateryear"].apply(lambda x: np.array([float(y) for y in re.findall(r"\d{4}", x)]))
df["mean_P_over_time"] = df["mean_P_over_time"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["mean_PET_over_time"] = df["mean_PET_over_time"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["mean_Q_over_time"] = df["mean_Q_over_time"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["aridity_over_time"] = df["aridity_over_time"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["cor_PET_P_over_time"] = df["cor_PET_P_over_time"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["sens_P_over_time_mr1"] = df["sens_P_over_time_mr1"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["sens_PET_over_time_mr1"] = df["sens_PET_over_time_mr1"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["R2_over_time_mr1"] = df["R2_over_time_mr1"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["sens_P_over_time_mr2"] = df["sens_P_over_time_mr2"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["sens_PET_over_time_mr2"] = df["sens_PET_over_time_mr2"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df["R2_over_time_mr2"] = df["R2_over_time_mr2"].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df['start_wateryear_first'] = df['start_wateryear'].apply(lambda x: x[0])
df['end_wateryear_last'] = df['end_wateryear'].apply(lambda x: x[-1])
df = df.reset_index()

# Budyko plot
fig, (ax1) = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
im1 = ax1.scatter(df["aridity_control"], df["mean_Q"]/df["mean_P"], s=5, alpha=0.8, c="tab:purple")
ax1.set_ylabel(r"$Q$/$P$ [-]")
ax1.set_xlim([0.1, 10])
ax1.set_xscale('log')
ax1.set_ylim([-0.2, 1.2])
# Common elements for both subplots
ax1.plot([0, 20], [1, 1], color='grey', linestyle='--', linewidth=1)
ax1.plot([1, 20], [0, 0], color='grey', linestyle='--', linewidth=1)
x = np.logspace(-1, 0, 100)
y = 1-x
ax1.plot(x, y, color='grey', linestyle='--', linewidth=1)
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
Q_vec = util_Turc.calculate_streamflow(P_vec, E0_vec, 2.2)
ax1.plot(E0_vec/P_vec, Q_vec/P_vec, color='black', linestyle='-', linewidth=2, label="n = 2.2")
Q_vec = util_Turc.calculate_streamflow(P_vec, E0_vec, 2)
#ax1.plot(E0_vec/P_vec, Q_vec/P_vec, color='black', linestyle=':', linewidth=2, label="n = 2")
#cbar = fig.colorbar(im1, ax=[ax1], label=' [-]', aspect=30)
ax1.set_xlabel(r"$E_p$/$P$ [-]")
#plt.legend()
plt.savefig(figures_path + 'budyko_plot.png', dpi=600)

# fit n value
df.loc[(df["country"] == 5) & (df["P_seasonality_index"] < 0), "country"] = 6

P = df["mean_P"]
PET = df["mean_PET"]
Q = df["mean_Q"]
def objective_function(n):
    Q_calc = util_Turc.calculate_streamflow(P, PET, n)
    return (abs(Q_calc - Q)).sum()
from scipy.optimize import minimize_scalar
res = minimize_scalar(objective_function, bounds=(0.1, 10), method='bounded')
print("Average Turc-Pike n: ", np.round(res.x, 2))
# fit Turc-Pike parameter n to each country and plot it
countries = df["country"].unique()
n_values = []
for country in countries:
    df_country = df[df["country"] == country]
    P = df_country["mean_P"]
    PET = df_country["mean_PET"]
    Q = df_country["mean_Q"]
    def objective_function(n):
        Q_calc = util_Turc.calculate_streamflow(P, PET, n)
        return (abs(Q_calc - Q)).sum()
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(objective_function, bounds=(0.1, 10), method='bounded')
    n_values.append(res.x)
    print("Average Turc-Pike n for country ", country, ": ", np.round(res.x, 2))

#
df_s = df.copy()
df_s = df_s.sort_values(by=["cor_PET_P"])
n_pos = np.sum(df_s["cor_PET_P"] > 0)

# Budyko plot with fitted n per country
country_codes = sorted(set(df["country"]))  # e.g. [2,3,4,5]
n_countries = len(country_codes)
colors = plt.cm.magma(np.linspace(0, 1, n_countries+1))
country_colors = dict(zip(country_codes, colors))

# Scatter with consistent country colors
fig, (ax1) = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
for country in country_codes:
    mask = df["country"] == country
    color = country_colors[country]
    ax1.scatter(df.loc[mask, "aridity_control"],
                df.loc[mask, "mean_Q"]/df.loc[mask, "mean_P"],
                s=5, alpha=0.8, c=[color]) #, label=f"Country {country}"
# Common Budyko elements
ax1.set_ylabel(r"$Q$/$P$ [-]")
ax1.set_xlim([0.1, 10])
ax1.set_xscale('log')
ax1.set_ylim([-0.2, 1.2])
ax1.plot([0, 20], [1, 1], color='grey', linestyle='--', linewidth=1)
ax1.plot([1, 20], [0, 0], color='grey', linestyle='--', linewidth=1)
x = np.logspace(-1, 0, 100)
y = 1-x
ax1.plot(x, y, color='grey', linestyle='--', linewidth=1)
# Plot fitted Turc curves using optimized n values
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
country_idx = {country: i for i, country in enumerate(country_codes)}
for i, country in enumerate(country_codes):
    n_country = n_values[country_idx[country]]  # Use optimized n
    Q_vec = util_Turc.calculate_streamflow(P_vec, E0_vec, n_country)
    color = country_colors[country]
    ax1.plot(E0_vec/P_vec, Q_vec/P_vec, linestyle='-', linewidth=2, color="white", alpha=1)
    ax1.plot(E0_vec/P_vec, Q_vec/P_vec, linestyle='-', linewidth=1, color=color, label=f"n={n_country:.1f} (country {country})", alpha=1)
ax1.set_xlabel(r"$E_p$/$P$ [-]")
#ax1.legend()#loc='lower right')
plt.savefig(figures_path + 'budyko_plot_fitted_n_per_country.png', dpi=600, bbox_inches='tight')

# Budyko plot with 1 country per subpanel (2x2 grid)
country_codes = sorted(set(df["country"]))  # Assumes exactly 4 countries
colors = plt.cm.magma(np.linspace(0, 1, len(country_codes) + 1))
country_colors = dict(zip(country_codes, colors))
fig, axs = plt.subplots(3, 2, figsize=(8, 5), constrained_layout=True, sharex=True, sharey=True)
axs = axs.ravel()
# Common Budyko elements for all panels
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
x = np.logspace(-1, 0, 100)
y = 1 - x
for i, country in enumerate(country_codes):
    ax = axs[i]
    # Scatter only THIS country
    mask = df["country"] == country
    color = country_colors[country]
    ax.scatter(df.loc[mask, "aridity_control"],
               df.loc[mask, "mean_Q"] / df.loc[mask, "mean_P"],
               s=5, alpha=0.8, c=[color], label=f"Country {country}")
    # Budyko reference lines
    ax.set_xlim([0.1, 10])
    ax.set_xscale('log')
    ax.set_ylim([-0.2, 1.2])
    ax.plot([0, 20], [1, 1], color='grey', linestyle='--', linewidth=1)
    ax.plot([1, 20], [0, 0], color='grey', linestyle='--', linewidth=1)
    ax.plot(x, y, color='grey', linestyle='--', linewidth=1)
    # Fitted Turc curve for THIS country only
    country_idx = {c: j for j, c in enumerate(country_codes)}
    n_country = n_values[country_idx[country]]
    Q_vec = util_Turc.calculate_streamflow(P_vec, E0_vec, n_country)
    ax.plot(E0_vec / P_vec, Q_vec / P_vec, linestyle='-', linewidth=2,
            color="white", alpha=1)
    ax.plot(E0_vec / P_vec, Q_vec / P_vec, linestyle='-', linewidth=2,
            color=color, label=f"n={n_country:.1f}")
    #ax.legend()
# Outer labels
for ax in axs[2:]:
    ax.set_xlabel(r"$E_p$/$P$ [-]")
for ax in axs[::2]:
    ax.set_ylabel(r"$Q$/$P$ [-]")
plt.savefig(figures_path + 'budyko_plot_one_country_per_panel.png', dpi=600, bbox_inches='tight')

# plot sensitivity as function of aridity and compare to theoretical sensitivity
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True, sharey=False)
ax = ax1
for country in country_codes:
    mask = df["country"] == country
    color = country_colors[country]
    ax.scatter(df.loc[mask, "aridity_control"],
                df.loc[mask, "sens_P_mr1"],
                s=5, alpha=0.8, c=[color]) #, label=f"Country {country}"
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
dQdP, dQdE0 = util_Turc.calculate_sensitivities(P_vec, E0_vec, n)
ax.plot(E0_vec/P_vec, dQdP, color='white', linestyle='-', linewidth=3, label="dQ/dP (theory)")
ax.plot(E0_vec/P_vec, dQdP, color='grey', linestyle='-', linewidth=2, label="dQ/dP (theory)")
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.set_ylabel(r"$s_P$ [-]")
ax.set_xscale('log')
ax.set_xlim([0.1, 10])
ax.set_ylim([-0.5, 1.5])
ax = ax2
for country in country_codes:
    mask = df["country"] == country
    color = country_colors[country]
    ax.scatter(df.loc[mask, "aridity_control"],
                df.loc[mask, "sens_PET_mr1"],
                s=5, alpha=0.8, c=[color]) #, label=f"Country {country}"ax.plot(E0_vec/P_vec, dQdE0, color='white', linestyle='-', linewidth=3, label="dQ/dPET (theory)")
ax.plot(E0_vec/P_vec, dQdE0, color='grey', linestyle='-', linewidth=2, label="dQ/dPET (theory)")
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.set_xlabel(r"$E_p$/$P$ [-]")
ax.set_ylabel(r"$s_{Ep}$ [-]")
ax.set_xscale('log')
ax.set_xlim([0.1, 10])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_func))
ax.set_ylim([-1.5, 0.5])
fig.tight_layout()
plt.savefig(figures_path + 'sensitivity_aridity_per_country.png', dpi=600)

# histogram of cov_P_PET
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(df["cor_PET_P"], bins=np.linspace(-1, 1, 100), alpha=0.7, color='grey')
ax.set_xlabel(r"Correlation between $P$ and $E_p$ [-]")
ax.set_ylabel("Count")
ax.set_xlim([-1, 1])
fig.tight_layout()
plt.savefig(figures_path + 'cor_P_PET_histogram.png', dpi=600)
print("Average correlation between P and PET: ", np.round(np.mean(df["cor_PET_P"]), 2))

# std of annual values divided by mean
df["std_P"] = df["annual_P"].apply(lambda x: np.std(x))
df["std_PET"] = df["annual_PET"].apply(lambda x: np.std(x))
df["std_Q"] = df["annual_Q"].apply(lambda x: np.std(x))
df["cv_P"] = df["std_P"] / df["mean_P"]
df["cv_PET"] = df["std_PET"] / df["mean_PET"]
df["cv_Q"] = df["std_Q"] / df["mean_Q"]
print("Median CV P: ", np.round(np.median(df["cv_P"]), 2))
print("Median CV PET: ", np.round(np.median(df["cv_PET"]), 2))
print("Median CV Q: ", np.round(np.median(df["cv_Q"]), 2))
# plot histograms in one figure
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(df["cv_P"], bins=np.linspace(0, 1, 100), alpha=0.7, color='tab:blue', label=r'$P$')
ax.hist(df["cv_PET"], bins=np.linspace(0, 1, 100), alpha=0.7, color='tab:orange', label=r'$E_p$')
ax.hist(df["cv_Q"], bins=np.linspace(0, 1, 100), alpha=0.7, color='tab:purple', label=r'$Q$')
ax.set_xlabel("Coefficient of variation of annual water fluxes [-]")
ax.set_ylabel("Count")
ax.set_xlim([0, 1])
ax.legend()
fig.tight_layout()
plt.savefig(figures_path + 'CoV_P_PET_Q_histogram.png', dpi=600)

# calculate Pearson and Spearman rank correlation between annual_P and annual_PET for all catchments and print it
cor_P_PET_pearson = []
cor_P_PET_spearman = []
for i in range(len(df)):
    series_p = pd.Series(df.loc[i, "annual_P"])
    series_pet = pd.Series(df.loc[i, "annual_PET"])
    pearson_corr = series_p.corr(series_pet, method='pearson')
    spearman_corr = series_p.corr(series_pet, method='spearman')
    cor_P_PET_pearson.append(pearson_corr)
    cor_P_PET_spearman.append(spearman_corr)
print("Average Pearson correlation between annual P and annual PET: ", np.round(np.nanmean(cor_P_PET_pearson), 2))
print("Average Spearman correlation between annual P and annual PET: ", np.round(np.nanmean(cor_P_PET_spearman), 2))

# extract catchments with ids = [73011, 'A2390531', 'DE810570', 2109]
ids = ['camels_02027000', 'camelsgb_73005', 'camelsde_DE911260', 'camelsaus_616216']
#ids = df["gauge_id"]
for id in ids:
    highlight = df[df["gauge_id"] == id]
    print(highlight[["gauge_name"]].round(2))
    print(highlight[["gauge_id", "cv_P", "cv_PET", "cv_Q"]].round(2))
    print(highlight[["gauge_id_native", "aridity_control", "sens_P_mr1", "sens_PET_mr1", "cor_PET_P"]].round(2))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
    ax = ax1
    ax.scatter(highlight["annual_P"].values[0]*365, highlight["annual_Q"].values[0]*365, s=25, c="tab:blue", alpha=0.8, lw=0)
    ax.set_xlabel(r"$P$ [mm/y]")
    ax.set_ylabel(r"$Q$ [mm/y]")
    ax = ax2
    ax.scatter(highlight["annual_PET"].values[0]*365, highlight["annual_Q"].values[0]*365, s=25, c="tab:orange", alpha=0.8, lw=0)
    ax.set_xlabel(r"$E_p$ [mm/y]")
    ax.set_ylabel(r"$Q$ [mm/y]")
    plt.tight_layout()
    plt.savefig(figures_path + 'example_plots_P_PET_Q/P_PET_vs_Q_' + id + '.png', dpi=600)
    plt.close()

# comparison between averaging and non-averaging methods
# compare methods
fig = plt.figure(figsize=(3.5, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_P_mr1"], df["sens_P_avg_mr1"], s=10, c='tab:blue', alpha=0.8, lw=0)
axes.set_xlabel(r"$s_P$ Mult. Reg. #1 [-]")
axes.set_ylabel(r"$s_P$ Mult. Reg. #1 5y average [-]")
axes.set_xlim([-0.1, 1.4])
axes.set_ylim([-0.1, 1.4])
axes.plot([-1, 3], [-1, 3], color='grey', linestyle='--', linewidth=1)
#plt.show()
plt.savefig(figures_path + 'sensitivity_comparison_P_averaging.png', dpi=600)
cor_P = df["sens_P_mr1"].corr(df["sens_P_avg_mr1"], method='spearman')
print("Correlation between dQ/dP #1 and dQ/dP #1 5y average: ", np.round(cor_P, 2))

# compare methods
fig = plt.figure(figsize=(3.5, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_PET_mr1"], df["sens_PET_avg_mr1"], s=10, c='tab:orange', alpha=0.8, lw=0) #c=df["cor_PET_P"]
axes.set_xlabel(r"$s_{Ep}$ Mult. Reg. #1 [-]")
axes.set_ylabel(r"$s_{Ep}$ Mult. Reg. #1 5y average [-]")
axes.set_xlim([-1.5, 1.])
axes.set_ylim([-1.5, 1.])
axes.plot([-2, 3], [-2, 3], color='grey', linestyle='--', linewidth=1)
#plt.show()
plt.savefig(figures_path + 'sensitivity_comparison_PET_averaging.png', dpi=600)
# print correlation
cor_PET = df["sens_PET_mr1"].corr(df["sens_PET_avg_mr1"], method='spearman')
print("Correlation between dQ/dPET #1 and dQ/dPET #1 5y average: ", np.round(cor_PET, 2))

# plot histograms of p values
fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
ax[0].hist(df["pval_sens_P_mr1"], bins=np.linspace(0, 1, 100), alpha=0.7, color='tab:blue')
ax[0].set_xlabel(r"$p$-value $s_{P}$ Mult. Reg. #1 [-]")
ax[0].set_ylabel("Count")
ax[0].set_xlim([0, 1])
ax[1].hist(df["pval_sens_PET_mr1"], bins=np.linspace(0, 1, 100), alpha=0.7, color='tab:orange')
ax[1].set_xlabel(r"$p$-value $s_{Ep}$ Mult. Reg. #1 [-]")
ax[1].set_ylabel("Count")
ax[1].set_xlim([0, 1])
plt.savefig(figures_path + 'p_value_histograms_mr1.png', dpi=600)
print("Number of catchments with p-value < 0.05 for dQ/dP #1: ", np.sum(df["pval_sens_P_mr1"] < 0.05))
print("Number of catchments with p-value < 0.05 for dQ/dPET #1: ", np.sum(df["pval_sens_PET_mr1"] < 0.05))

# plot histograms of p values
fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
ax[0].hist(df["pval_sens_P_mr2"], bins=np.linspace(0, 1, 100), alpha=0.7, color='tab:blue')
ax[0].set_xlabel(r"$p$-value $s_{P}$ Mult. Reg. #2 [-]")
ax[0].set_ylabel("Count")
ax[0].set_xlim([0, 1])
ax[1].hist(df["pval_sens_PET_mr2"], bins=np.linspace(0, 1, 100), alpha=0.7, color='tab:orange')
ax[1].set_xlabel(r"$p$-value $s_{Ep}$ Mult. Reg. #2 [-]")
ax[1].set_ylabel("Count")
ax[1].set_xlim([0, 1])
plt.savefig(figures_path + 'p_value_histograms_mr1.png', dpi=600)
print("Number of catchments with p-value < 0.05 for dQ/dP #2: ", np.sum(df["pval_sens_P_mr2"] < 0.05))
print("Number of catchments with p-value < 0.05 for dQ/dPET #2: ", np.sum(df["pval_sens_PET_mr2"] < 0.05))

# plot p-value of #2 vs sensitivity of #1
fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
ax[0].scatter(df["pval_sens_P_mr1"], df["sens_P_mr1"], s=10, c="tab:blue", alpha=0.8, lw=0)
ax[0].set_xlabel(r"$p$-value $s_{P}$ Mult. Reg. #1 [-]")
ax[0].set_ylabel(r"Sensitivity $s_{P}$ Mult. Reg. #1 [-]")
ax[0].set_xlim([-0.2, 1.2])
ax[0].set_ylim([-0.5, 1.5])
ax[0].text(0.95, 0.95, f"p < 0.05: {np.round(np.sum(df['pval_sens_P_mr1'] < 0.05)/len(df)*100)} %",
              fontsize=10, ha='right', va='top', transform=ax[0].transAxes)
ax[1].scatter(df["pval_sens_PET_mr1"], df["sens_PET_mr1"], s=10, c="tab:orange", alpha=0.8, lw=0)
ax[1].set_xlabel(r"$p$-value $s_{Ep}$ Mult. Reg. #1 [-]")
ax[1].set_ylabel(r"Sensitivity $s_{Ep}$ Mult. Reg. #1 [-]")
ax[1].set_xlim([-0.2, 1.2])
ax[1].set_ylim([-1.5, 0.5])
ax[1].text(0.95, 0.95, f"p < 0.05: {np.round(np.sum(df['pval_sens_PET_mr1'] < 0.05)/len(df)*100)} %",
                fontsize=10, ha='right', va='top', transform=ax[1].transAxes)
plt.savefig(figures_path + 'p_value_sensitivity_mr1.png', dpi=600)

# plot p-value of #2 vs sensitivity of #2
fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
ax[0].scatter(df["pval_sens_P_mr2"], df["sens_P_mr2"], s=10, c="tab:blue", alpha=0.8, lw=0)
ax[0].set_xlabel(r"$p$-value $s_{P}$ Mult. Reg. #2 [-]")
ax[0].set_ylabel(r"Sensitivity $s_{P}$ Mult. Reg. #2 [-]")
ax[0].set_xlim([-0.2, 1.2])
ax[0].set_ylim([-0.5, 1.5])
ax[0].text(0.95, 0.95, f"p < 0.05: {np.round(np.sum(df['pval_sens_P_mr2'] < 0.05)/len(df)*100)} %",
           fontsize=10, ha='right', va='top', transform=ax[0].transAxes)
ax[1].scatter(df["pval_sens_PET_mr2"], df["sens_PET_mr2"], s=10, c="tab:orange", alpha=0.8, lw=0)
ax[1].set_xlabel(r"$p$-value $s_{Ep}$ Mult. Reg. #2 [-]")
ax[1].set_ylabel(r"Sensitivity $s_{Ep}$ Mult. Reg. #2 [-]")
ax[1].set_xlim([-0.2, 1.2])
ax[1].set_ylim([-3, 3])
ax[1].text(0.95, 0.95, f"p < 0.05: {np.round(np.sum(df['pval_sens_PET_mr2'] < 0.05)/len(df)*100)} %",
              fontsize=10, ha='right', va='top', transform=ax[1].transAxes)
plt.savefig(figures_path + 'p_value_sensitivity_mr2.png', dpi=600)

# plot p values vs R²
fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
ax[0].scatter(df["pval_sens_P_mr1"], df["R2_mr1"], s=5, c="tab:blue", alpha=0.8, lw=0)
ax[0].set_xlabel(r"$p$-value $s_{P}$ Mult. Reg. #1 [-]")
ax[0].set_ylabel(r"$R^2$ Mult. Reg. #1 [-]")
# set x axis to log scale
ax[0].set_xscale('log')
ax[0].set_ylim([0, 1])
ax[1].scatter(df["pval_sens_PET_mr1"], df["R2_mr1"], s=5, c="tab:orange", alpha=0.8, lw=0)
ax[1].set_xlabel(r"$p$-value $s_{Ep}$ Mult. Reg. #1 [-]")
ax[1].set_ylabel(r"$R^2$ Mult. Reg. #1 [-]")
#ax[0].set_xlim([0, 0.1])
ax[1].set_xscale('log')
ax[1].set_ylim([0, 1])
plt.savefig(figures_path + 'p_value_R2.png', dpi=600)

# compare methods
fig = plt.figure(figsize=(3, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_P_mr1"], df["sens_P_mr2"], s=5, c=df["cor_PET_P"], alpha=0.8, lw=0, vmin=-0.7, vmax=0.1, cmap='magma')
axes.set_xlabel(r"$s_P$ Mult. Reg. #1 [-]")
axes.set_ylabel(r"$s_P$  Mult. Reg. #2 [-]")
axes.set_xlim([-0.1, 1.4])
axes.set_ylim([-0.1, 1.4])
axes.plot([-1, 3], [-1, 3], color='grey', linestyle='--', linewidth=1)
#plt.grid()
#cbar = plt.colorbar(im, ax=axes)
#cbar.set_label('Cor(P,PET) [-]')
plt.savefig(figures_path + 'sensitivity_comparison_P_cor.png', dpi=600)
cor_P = df["sens_P_mr1"].corr(df["sens_P_mr2"], method='spearman')
print("Correlation between dQ/dP #1 and dQ/dP #2: ", np.round(cor_P, 2))

# compare methods
fig = plt.figure(figsize=(3, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_PET_mr1"], df["sens_PET_mr2"], s=5, c=df["cor_PET_P"], alpha=0.8, lw=0, vmin=-0.7, vmax=0.1, cmap='magma') #c=df["cor_PET_P"]
axes.set_xlabel(r"$s_{Ep}$ Mult. Reg. #1 [-]")
axes.set_ylabel(r"$s_{Ep}$  Mult. Reg. #2 [-]")
axes.set_xlim([-2, 3])
axes.set_ylim([-2, 3])
axes.plot([-2, 3], [-2, 3], color='grey', linestyle='--', linewidth=1)
#plt.grid()
#cbar = plt.colorbar(im, ax=axes)
#cbar.set_label('Cor(P,PET) [-]')
plt.savefig(figures_path + 'sensitivity_comparison_PET_cor.png', dpi=600)
# print correlation
cor_PET = df["sens_PET_mr1"].corr(df["sens_PET_mr2"], method='spearman')
print("Correlation between dQ/dPET #1 and dQ/dPET #2: ", np.round(cor_PET, 2))

# compare methods
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["cor_PET_P"], df["sens_PET_mr2"], s=10, c=df["sens_PET_mr1"], alpha=0.8, lw=0, vmin=-1, vmax=0, cmap='magma') #c=df["cor_PET_P"]
axes.set_xlabel(r"cor(P,PET) [-]")
axes.set_ylabel(r"$s_{Ep}$  Mult. Reg. #2 [-]")
axes.set_xlim([-1, 0.5])
axes.set_ylim([-2, 4])
#axes.plot([-2, 3], [-2, 3], color='grey', linestyle='--', linewidth=1)
#plt.grid()
cbar = plt.colorbar(im, ax=axes)
plt.show()
#cbar.set_label('Cor(P,PET) [-]')
#plt.savefig(figures_path + 'sens_P_vs_cor.png', dpi=600)
# print correlation

# sensitivity and different variables
pf.plot_sensitivity_aridity_variable(df, n, "R2_mr1", 0, 1, "magma_r", figures_path)

### storage sensitivity

# plot histogram of sensitivities
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(df["sens_P_storage_mr1"], bins=np.linspace(-1.5,1.5,100), alpha=0.7, color='tab:blue', label='dQ/dP')
ax.hist(df["sens_PET_storage_mr1"], bins=np.linspace(-1.5,1.5,100), alpha=0.7, color='tab:orange', label='dQ/dPET')
ax.hist(df["sens_Q_storage_mr1"], bins=np.linspace(-1.5,1.5,100), alpha=0.7, color='tab:purple', label='dQ/dQlag1')
ax.set_xlabel("Sensitivity [-]")
ax.set_ylabel("Count")
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([0, 125])
fig.tight_layout()
plt.savefig(figures_path + 'storage_sensitivity_histogram.png', dpi=600)

# compare methods
fig = plt.figure(figsize=(3, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_P_mr1"], df["sens_P_storage_mr1"], s=5, c="tab:blue", alpha=0.8, lw=0, vmin=-0.5, vmax=0.0, cmap='magma')
axes.set_xlabel(r"$s_P$ Mult. Reg. #1 [-]")
axes.set_ylabel(r"$s_P$ Storage Mult. Reg. #1 [-]")
axes.set_xlim([-0.1, 1.4])
axes.set_ylim([-0.1, 1.4])
axes.plot([-1, 3], [-1, 3], color='grey', linestyle='--', linewidth=1)
#plt.grid()
#cbar = plt.colorbar(im, ax=axes)
#cbar.set_label('Cor(P,PET) [-]')
plt.savefig(figures_path + 'sensitivity_comparison_storage_P.png', dpi=600)
cor_P = df["sens_P_mr1"].corr(df["sens_P_storage_mr1"], method='spearman')
print("Correlation between dQ/dP #1 and dQ/dP #1 with storage: ", np.round(cor_P, 2))

# compare methods
fig = plt.figure(figsize=(3, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_PET_mr1"], df["sens_PET_storage_mr1"], s=5, c="tab:orange", alpha=0.8, lw=0, vmin=-0.5, vmax=0.0, cmap='magma') #c=df["cor_PET_P"]
axes.set_xlabel(r"$s_{Ep}$ Mult. Reg. #1 [-]")
axes.set_ylabel(r"$s_{Ep}$ Storage Mult. Reg. #1 [-]")
axes.set_xlim([-1.5, 1])
axes.set_ylim([-1.5, 1])
axes.plot([-2, 3], [-2, 3], color='grey', linestyle='--', linewidth=1)
#plt.grid()
#cbar = plt.colorbar(im, ax=axes)
#cbar.set_label('Cor(P,PET) [-]')
plt.savefig(figures_path + 'sensitivity_comparison_storage_PET.png', dpi=600)
# print correlation
cor_PET = df["sens_PET_mr1"].corr(df["sens_PET_storage_mr1"], method='spearman')
print("Correlation between dQ/dPET #1 and dQ/dPET #1 with storage: ", np.round(cor_PET, 2))

# compare R2
fig = plt.figure(figsize=(3, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["R2_mr1"], df["R2_storage_mr1"], s=5, c="black", alpha=0.8)
axes.set_xlabel(r"$R^2$ Mult. Reg. #1 [-]")
axes.set_ylabel(r"$R^2$ Mult. Reg. #1 with Storage [-]")
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
axes.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1)
plt.savefig(figures_path + 'R2_storage_comparison.png', dpi=600)
# print mean and median R2 for both
median_R2_mr1 = np.nanmedian(df["R2_mr1"])
median_R2_mr2 = np.nanmedian(df["R2_storage_mr1"])
print("Median R2 Mult. Reg. #1: ", np.round(median_R2_mr1, 2))
print("Median R2 Mult. Reg. #1 with Storage: ", np.round(median_R2_mr2, 2))

# plot sensitivity as function of aridity and compare to theoretical sensitivity
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 7.5), sharex=True, sharey=False)
ax = ax1
ax.scatter(df["aridity_control"], df["sens_P_storage_mr1"], s=5, c="tab:blue", alpha=0.8, lw=0, label="dQ/dP (data)")
P_vec = np.linspace(0.01, 10, 100)
E0_vec = np.linspace(10, 0.01, 100)
dQdP, dQdE0 = util_Turc.calculate_sensitivities(P_vec, E0_vec, n)
ax.plot(E0_vec/P_vec, dQdP, color='white', linestyle='-', linewidth=3, label="dQ/dP (theory)")
ax.plot(E0_vec/P_vec, dQdP, color='tab:blue', linestyle='-', linewidth=2, label="dQ/dP (theory)")
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.set_ylabel(r"$s_P$ [-]")
ax.set_xscale('log')
ax.set_xlim([0.1, 10])
ax.set_ylim([-0.5, 1.5])
ax = ax2
ax.scatter(df["aridity_control"], df["sens_PET_storage_mr1"], s=5, c="tab:orange", alpha=0.8, lw=0, label="dQ/dPET (data)")
ax.plot(E0_vec/P_vec, dQdE0, color='white', linestyle='-', linewidth=3, label="dQ/dPET (theory)")
ax.plot(E0_vec/P_vec, dQdE0, color='tab:orange', linestyle='-', linewidth=2, label="dQ/dPET (theory)")
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.set_ylabel(r"$s_{Ep}$ [-]")
ax.set_xscale('log')
ax.set_xlim([0.1, 10])
ax.set_ylim([-1.5, 0.5])
ax = ax3
ax.scatter(df["aridity_control"], df["sens_Q_storage_mr1"], s=5, c="tab:brown", alpha=0.8, lw=0, label="dQ/dQ(t-1) (data)")
#ax.plot(E0_vec/P_vec, dQdE0, color='tab:purple', linestyle='--', linewidth=2, label="dQ/dPET (theory)")
ax.axhline(0, color='grey', linestyle='--', linewidth=1)
ax.set_xlabel(r"$E_p$/$P$ [-]")
ax.set_ylabel(r"$s_{Q(t-1)}$ [-]")
ax.set_xscale('log')
ax.set_xlim([0.1, 10])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_func))
ax.set_ylim([-0.5, 1.5])
fig.tight_layout()
plt.savefig(figures_path + 'storage_sensitivity_aridity_panels.png', dpi=600)

# compare storage sensitivity to BFI
fig = plt.figure(figsize=(3, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df["sens_Q_storage_mr1"], df["BFI"], s=5, c="black", alpha=0.8)
axes.set_xlabel(r"$s_{Q(t-1)}$ [-]")
axes.set_ylabel(r"BFI [-]")
axes.set_xlim([-0.2, 0.8])
axes.set_ylim([0, 1])
axes.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1)
plt.savefig(figures_path + 'storage_sensivity_vs_BFI.png', dpi=600)
cor_BFI = df["sens_Q_storage_mr1"].corr(df["BFI"], method='spearman')
print("Correlation between dQ/dQ(t-1) and BFI: ", np.round(cor_BFI, 2))

# identify 4 example catchments
# lies on theoretical curve: camelsgb_73011
# high snow fraction: lamah_202697
# low P sensitivity and high BFI: camelsde_DEE10100
# medium P sensitivity and low BFI: camels_01516500
#highlight_ids = [73011, 'A2390531', 'DE810570', 2109]
#highlight = df[df["gauge_id"].isin(highlight_ids)]
#ax1.scatter(highlight["aridity_control"], highlight["sens_P_mr2"], s=50, c="tab:orange", alpha=0.8, lw=0)
#ax2.scatter(highlight["aridity_control"], highlight["sens_PET_mr1"], s=50, c="tab:orange", alpha=0.8, lw=0)


# make histogram of seasonality_index per country
# make smoothed density plots (KDE) of seasonality_index per country instead of histogram
country_codes = sorted(set(df["country"]))  # e.g. [2,3,4,5]
n_countries = len(country_codes)
colors = plt.cm.magma(np.linspace(0, 1, n_countries + 1))
country_colors = dict(zip(country_codes, colors))

fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
for country in country_codes:
    mask = df["country"] == country
    data = df.loc[mask, "seasonality_index"].values
    color = country_colors[country]
    kde = gaussian_kde(data)
    x = np.linspace(0, 2, 500)
    y = kde(x)
    #scale_factor = len(data)   # Adjust to visually match histogram density
    #ax.plot(x, y, color=color, linewidth=2, label=country)
    ax.fill_between(x, 0, y, color=color, alpha=0.5)
    ax.set_xlabel(r"Seasonality Index [-]")
ax.set_ylabel(r"Density [-]")
ax.set_xlim([0, 2])
ax.legend(title="Country", labels=['USA', 'GB', 'DE', 'AUS'])
plt.savefig(figures_path + 'seasonality_index_kde_countries.png', dpi=600)

fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
for country in country_codes:
    mask = df["country"] == country
    data = df.loc[mask, "aridity_control"].values
    color = country_colors[country]
    kde = gaussian_kde(data)
    x = np.linspace(0, 4, 500)
    y = kde(x)
    #scale_factor = len(data)   # Adjust to visually match histogram density
    #ax.plot(x, y, color=color, linewidth=2, label=country)
    ax.fill_between(x, 0, y, color=color, alpha=0.5)
    ax.set_xlabel(r"Aridity Index [-]")
ax.set_ylabel(r"Density [-]")
ax.set_xlim([0, 3])
#ax.set_xscale('log')
ax.legend(title="Country", labels=['USA', 'GB', 'DE', 'AUS'])
plt.savefig(figures_path + 'aridity_index_kde_countries.png', dpi=600)

fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
for country in country_codes:
    mask = df["country"] == country
    data = df.loc[mask, "BFI"].values
    color = country_colors[country]
    kde = gaussian_kde(data)
    x = np.linspace(0, 1, 500)
    y = kde(x)
    #scale_factor = len(data)   # Adjust to visually match histogram density
    #ax.plot(x, y, color=color, linewidth=2, label=country)
    ax.fill_between(x, 0, y, color=color, alpha=0.5)
    ax.set_xlabel(r"BFI [-]")
ax.set_ylabel(r"Density [-]")
ax.set_xlim([0, 1])
ax.legend(title="Country", labels=['USA', 'GB', 'DE', 'AUS'])
plt.savefig(figures_path + 'bfi_kde_countries.png', dpi=600)

# make histogram of BFI per country
country_codes = sorted(set(df["country"]))  # e.g. [2,3,4,5]
n_countries = len(country_codes)
colors = plt.cm.magma(np.linspace(0, 1, n_countries+1))
country_colors = dict(zip(country_codes, colors))
fig, (ax) = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
for country in country_codes:
    mask = df["country"] == country
    color = country_colors[country]
    ax.hist(df.loc[mask, "BFI"],
             bins=np.linspace(0, 1, 21),
             alpha=0.7, color=color, label=country)
ax.set_xlabel(r"BFI [-]")
ax.set_ylabel(r"Count")
ax.set_xlim([0, 1])
ax.legend(title="Country", labels=['USA', 'GB', 'DE', 'AUS'])
plt.savefig(figures_path + 'bfi_histogram_countries.png', dpi=600)

# just for the three subsets for trend analysis
colors = plt.cm.magma(np.linspace(0.5, 1, 4))
country_colors = dict(zip([4,5,6], colors))

df_filtered = df.copy()
df_filtered = df_filtered[df_filtered["len_years"] > 50]
#df_filtered.loc[(df_filtered["country"] == 5) & (df_filtered["P_seasonality_index"] > 0), "country"] = 6
fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
for country in [4,5,6]:
    mask = df_filtered["country"] == country
    data = df_filtered.loc[mask, "BFI"].values
    color = country_colors[country]
    kde = gaussian_kde(data)
    x = np.linspace(0, 1, 500)
    y = kde(x)
    #scale_factor = len(data)   # Adjust to visually match histogram density
    #ax.plot(x, y, color=color, linewidth=2, label=country)
    ax.fill_between(x, 0, y, color=color, alpha=0.5)
    ax.set_xlabel(r"BFI [-]")
ax.set_ylabel(r"Density [-]")
ax.set_xlim([0, 1])
ax.legend(title="Country", labels=['DE', 'AUS $P_S$>0', 'AUS $P_S$<0'])
plt.savefig(figures_path + 'bfi_kde_countries_filtered.png', dpi=600)

df_filtered = df.copy()
df_filtered = df_filtered[df_filtered["len_years"] > 50]
fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
for country in [4,5,6]:
    mask = df_filtered["country"] == country
    data = df_filtered.loc[mask, "P_seasonality_index"].values
    color = country_colors[country]
    kde = gaussian_kde(data)
    x = np.linspace(-1.5, 1.5, 500)
    y = kde(x)
    #scale_factor = len(data)   # Adjust to visually match histogram density
    #ax.plot(x, y, color=color, linewidth=2, label=country)
    ax.fill_between(x, 0, y, color=color, alpha=0.5)
    ax.set_xlabel(r"$P_S$ [-]")
ax.set_ylabel(r"Density [-]")
ax.set_xlim([-1.5, 1.5])
ax.legend(title="Country", labels=['DE', 'AUS $P_S$>0', 'AUS $P_S$<0'])
plt.savefig(figures_path + 'Ps_kde_countries_filtered.png', dpi=600)

df_filtered = df.copy()
df_filtered = df_filtered[df_filtered["len_years"] > 50]
fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
for country in [4,5,6]:
    mask = df_filtered["country"] == country
    data = df_filtered.loc[mask, "frac_snow_control"].values
    color = country_colors[country]
    kde = gaussian_kde(data)
    x = np.linspace(0, 1, 500)
    y = kde(x)
    #scale_factor = len(data)   # Adjust to visually match histogram density
    #ax.plot(x, y, color=color, linewidth=2, label=country)
    ax.fill_between(x, 0, y, color=color, alpha=0.5)
    ax.set_xlabel(r"$f_S$ [-]")
ax.set_ylabel(r"Density [-]")
ax.set_xlim([0, 1])
ax.legend(title="Country", labels=['DE', 'AUS $P_S$>0', 'AUS $P_S$<0'])
plt.savefig(figures_path + 'fs_kde_countries_filtered.png', dpi=600)
