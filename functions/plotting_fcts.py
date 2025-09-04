import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import rasterio
from rasterio.plot import plotting_extent, show
import geopandas as gpd
from pyproj import Transformer
from mpl_toolkits.basemap import Basemap
from functions import util_Turc
import matplotlib.ticker as ticker
def fmt_func(x, pos):
    return f"{round(x, 1):.1f}"

def partial_regression_plot(y, x1, x2, ax):
    """
    Plot partial regressions.

    Parameters:
    y (pd.Series): Dependent variable
    x1 (pd.Series): Independent variable of interest
    x2 (pd.DataFrame): Other independent variables to control for
    ax (matplotlib.axes.Axes): Axes object to plot on
    """

    # Prepare variables
    y = y.values.reshape(-1, 1)
    x1 = x1.values.reshape(-1, 1)
    x2 = x2.values

    # Regress y on x2 (control variables)
    model_y = LinearRegression(fit_intercept=False)
    model_y.fit(x2, y)
    res_y = y - model_y.predict(x2)

    # Regress x1 on x2 (control variables)
    model_x = LinearRegression(fit_intercept=False)
    model_x.fit(x2, x1)
    res_x = x1 - model_x.predict(x2)

    # Plot residuals
    ax.scatter(res_x, res_y, c='tab:blue', alpha=0.8, s=25)
    ax.set_xlabel('X residuals')
    ax.set_ylabel('Y residuals')

    # Add regression line
    model = LinearRegression(fit_intercept=False)
    model.fit(res_x, res_y)

    # Generate points for the regression line
    x_line = np.linspace(res_x.min(), res_x.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    # Plot regression line
    ax.plot(x_line, y_line, color='tab:orange')

    # Add R-squared value to the plot
    r_squared = model.score(res_x, res_y)
    ax.text(0.05, 0.95, f'R² = {r_squared:.2f}', transform=ax.transAxes,
            verticalalignment='top')

def plot_map(df, variable,
             cmap='viridis', vmin=None, vmax=None,
             cbar_label=None, figsize=(4, 4)):
    """
    Plots a geographic map with points colored by a specified variable.

    Parameters:
    df (DataFrame): Input dataframe containing geographic and variable data
    variable (str): Column name of the variable to visualize
    cmap (str): Matplotlib colormap name (default: 'Set1')
    vmin/vmax (float): Color scale limits (default: data min/max)
    cbar_label (str): Custom colorbar label (default: variable name)
    figsize (tuple): Figure dimensions (default: (4, 4))
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Initialize map
    m = Basemap(projection='robin', resolution='l',
                area_thresh=1000.0, lat_0=0, lon_0=0)

    # Draw map elements
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgrey', lake_color='white')
    m.drawmapboundary(fill_color='white')

    # Convert coordinates
    x, y = m(df["gauge_lon"].values, df["gauge_lat"].values)

    # Create scatter plot
    scatter = m.scatter(x, y, s=10, c=df[variable],
                        alpha=0.9, vmin=vmin, vmax=vmax,
                        cmap=cmap)

    # Configure colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02,
                        shrink=0.3, aspect=20)
    cbar_label = cbar_label if cbar_label else variable
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    # Set axis limits
    ax.set_xlim(np.min(x) * 0.99, np.max(x) * 1.01)
    ax.set_ylim(np.min(y) * 0.99, np.max(y) * 1.01)

    plt.tight_layout()
    plt.show()

def plot_map_with_dem(df, variable, dem_path,
                      cmap='viridis', vmin=0.0, vmax=0.5,
                      cbar_label=None, buffer=1, figsize=(10, 8),
                      dem_cmap='gray', dem_alpha=0.7):
    """
    Plots a variable on a digital elevation model (DEM) background.

    Parameters:
    df (pd.DataFrame): Input dataframe with geographic coordinates
    variable (str): Column name to visualize
    dem_path (str): Path to DEM GeoTIFF file
    cmap (str): Colormap for variable (default: 'viridis')
    vmin/vmax (float): Color scale limits (default: 0.0, 0.5)
    cbar_label (str): Colorbar label (default: variable name)
    buffer (float): Padding around data points in degrees (default: 1)
    figsize (tuple): Figure size (default: (10, 8))
    dem_cmap (str): DEM colormap (default: 'gray')
    dem_alpha (float): DEM transparency (default: 0.77)
    """
    # Load DEM and prepare visualization
    with rasterio.open(dem_path) as src:
        dem_array = src.read(1)
        dem_extent = plotting_extent(src)

        # Create coordinate transformer
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    # Prepare DEM colormap
    cmap_dem = plt.cm.get_cmap(dem_cmap)
    cmap_dem.set_bad('lightsteelblue')
    dem_masked = np.ma.masked_where(dem_array <= 0, dem_array)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot DEM background
    im = ax.imshow(dem_masked, cmap=cmap_dem, vmin=0, vmax=2000,
                   extent=dem_extent, alpha=dem_alpha)

    # Transform geographic coordinates to DEM's CRS
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['gauge_lon'], df['gauge_lat']),
        crs="EPSG:4326"
    )
    gdf = gdf.to_crs(src.crs)  # Match DEM coordinate system

    # Create scatter plot
    scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                         s=10, c=gdf[variable],
                         alpha=0.9, vmin=vmin, vmax=vmax,
                         cmap=cmap)

    # Configure colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02,
                        shrink=0.25, aspect=20)
    cbar_label = cbar_label if cbar_label else variable
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    # Set axis limits and labels
    # ax.set_xlim(gdf.geometry.x.min() - buffer,
    #            gdf.geometry.x.max() + buffer)
    # ax.set_ylim(gdf.geometry.y.min() - buffer,
    #            gdf.geometry.y.max() + buffer)

    ax.set_xlabel('Easting (m)' if src.crs.is_projected else 'Longitude')
    ax.set_ylabel('Northing (m)' if src.crs.is_projected else 'Latitude')

    plt.tight_layout()
    plt.show()

def plot_sensitivity_aridity_variable(df, n, var, vmin, vmax, cmap, figures_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), constrained_layout=True, sharex=True, sharey=False)
    im1 = ax1.scatter(df["aridity_control"], df["sens_P_mr1"], s=5, c=df[var], vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_ylabel(r"$s_P$ [-]")
    ax1.set_xlim([0.1, 10])
    ax1.set_xscale('log')
    ax1.set_ylim([-0.5, 1.5])
    im2 = ax2.scatter(df["aridity_control"], df["sens_PET_mr1"], s=5, c=df[var], vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_xlabel(r"$E_p$/$P$ [-]")
    ax2.set_ylabel(r"$s_{Ep}$ [-]")
    ax2.set_xlim([0.1, 10])
    ax2.set_xscale('log')
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_func))
    ax2.set_ylim([-1.5, 0.5])
    for ax in [ax1, ax2]:
        ax.plot([0.1, 20], [0, 0], color='grey', linestyle='--', linewidth=1)
        P_vec = np.linspace(0.01, 10, 100)
        E0_vec = np.linspace(10, 0.01, 100)
        dQdP, dQdE0 = util_Turc.calculate_sensitivities(P_vec, E0_vec, n)
        Q_vec = util_Turc.calculate_streamflow(P_vec, E0_vec, n)
        if ax == ax1:
            ax.plot(E0_vec / P_vec, dQdP, color='white', linestyle='-', linewidth=3)
            ax.plot(E0_vec / P_vec, dQdP, color='grey', linestyle='-', linewidth=2)
        else:
            ax.plot(E0_vec / P_vec, dQdE0, color='white', linestyle='-', linewidth=3)
            ax.plot(E0_vec / P_vec, dQdE0, color='grey', linestyle='-', linewidth=2)
    cbar = fig.colorbar(im1, ax=[ax1, ax2], label=var, aspect=30)
    plt.savefig(figures_path + 'sensitivity_aridity_' + var + '.png', dpi=600)

    df["dev_P"] = (df["sens_P_mr1"] - util_Turc.calculate_sensitivities(df["mean_P"], df["mean_PET"], n)[0])/df["sens_P_mr1"]
    df["dev_PET"] = (df["sens_PET_mr1"] - util_Turc.calculate_sensitivities(df["mean_P"], df["mean_PET"], n)[1])/df["sens_PET_mr1"]

    corr_P = df["dev_P"].corr(df[var], method='spearman')
    corr_PET = df["dev_PET"].corr(df[var], method='spearman')
    print("Correlation between relative deviation of P sensitivity and " + var + ": ", np.round(corr_P, 2))
    print("Correlation between relative deviation of PET sensitivity and " + var + ": ", np.round(corr_PET, 2))