import numpy as np
from scipy.odr import ODR, Model, RealData
from scipy.stats import t

# Define the linear function: with intercept
def linear_func_with_intercept(B, x):
    # B[0] = coeff for P, B[1] = coeff for PET, B[2] = intercept
    return B[0]*x[0] + B[1]*x[1] + B[2]

# Define the linear function: no intercept
def linear_func_no_intercept(B, x):
    # B[0] = coeff for P, B[1] = coeff for PET
    return B[0]*x[0] + B[1]*x[1]

def run_odr(df_art, method="with_intercept"):

    # Prepare data
    X = np.vstack([df_art["P"].values, df_art["PET"].values])  # shape (2, N)
    Y = df_art["Q"].values

    sx = np.ones_like(X)
    sy = np.ones_like(Y)

    data = RealData(X, Y, sx=sx, sy=sy)

    if method == "with_intercept":
        model = Model(linear_func_with_intercept)
        beta0 = [1.0, 1.0, 1.0]  # initial guess for P_coef, PET_coef, intercept
    elif method == "without_intercept":
        model = Model(linear_func_no_intercept)
        beta0 = [1.0, 1.0]  # initial guess for P_coef, PET_coef

    odr = ODR(data, model, beta0=beta0)
    out = odr.run()

    # Extract coefficients
    if method == "with_intercept":
        sens_P, sens_PET, intercept = out.beta
        p_std_err = out.sd_beta  # standard errors
        # Approximate p-values using t-distribution
        dof = len(Y) - 3  # N - number of parameters
        t_stats = np.abs(out.beta / p_std_err)
        p_values = 2 * (1 - t.cdf(t_stats, df=dof))  # two-sided
    elif method == "without_intercept":
        sens_P, sens_PET = out.beta
        intercept = None
        p_std_err = out.sd_beta
        dof = len(Y) - 2
        t_stats = np.abs(out.beta / p_std_err)
        p_values = 2 * (1 - t.cdf(t_stats, df=dof))  # two-sided

    # Calculate R2 for ODR
    fitted = out.y  # predicted values from fitted model
    ss_res = np.sum((Y - fitted) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2_odr = 1 - ss_res / ss_tot

    # Return coefficients, intercept, p_values, r2
    if method == "with_intercept":
        return sens_P, sens_PET, intercept, p_values, r2_odr
    elif method == "without_intercept":
        return sens_P, sens_PET, p_values, r2_odr
