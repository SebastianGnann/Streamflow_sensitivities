# function
library(lubridate)
library(dplyr)
library(lmodel2)

stepwise_regression_plot <- function(Q, P, PET, t, wateryear_start = 10, plot_results = TRUE, fit_intercept = FALSE) {
  
  t <- as.Date(t, format = "%Y-%m-%d")
  
  # Create dataframe and calculate wateryear
  df <- data.frame(t = t, Q = Q, P = P, PET = PET) %>%
    mutate(
      # Define the water year based on the starting month
      wateryear = ifelse(month(t) >= wateryear_start, year(t) + 1, year(t))
    )
  
  # Calculate annual averages for each water year
  df_annual <- df %>%
    group_by(wateryear) %>%
    # Filter water years with at least 365 non-NA values for Q, P, and PET
    filter(sum(!is.na(Q)) >= 365 & sum(!is.na(P)) >= 365 & sum(!is.na(PET)) >= 365) %>%
    # Summarize annual averages
    summarise(
      Q = sum(Q, na.rm = FALSE)/365,
      P = sum(P, na.rm = FALSE)/365,
      PET = sum(PET, na.rm = FALSE)/365,
      .groups = "drop" # Ungroup after summarizing
    )
  
  # Method 1: Non-centered regression
  model1 <- lm(Q ~ 0 + P + PET, data = df_annual) # -1
  sens_p1 <- round(coef(model1)["P"], 2)
  sens_pet1 <- round(coef(model1)["PET"], 2)
  cat(paste("Method 1: Sensitivity to P:", sens_p1, "Sensitivity to PET:", sens_pet1, "\n"))
  
  # Method 2: Centered regression
  df_centered <- df_annual %>%
    mutate(across(c(Q, P, PET), ~ .x - mean(.x)))
  model2 <- lm(Q ~ P + PET, data = df_centered)
  #model2 <- lm(log(Q) ~ log(P) + log(PET), data = df_annual)
  sens_p2 <- round(coef(model2)["P"], 2) #*mean(df_annual$Q)/mean(df_annual$P)
  sens_pet2 <- round(coef(model2)["PET"], 2) #*mean(df_annual$Q)/mean(df_annual$PET)
  cat(paste("Method 2: Sensitivity to P:", sens_p2, "Sensitivity to PET:", sens_pet2, "\n"))
  
  # Residual calculations
  model_P_Q <- lm(Q ~ 0 + P, data = df_centered)
  model_PET_Q <- lm(Q ~ 0 + PET, data = df_centered)
  model_P_PET <- lm(PET ~ 0 + P, data = df_centered)
  model_PET_P <- lm(P ~ 0 + PET, data = df_centered)
  
  resid_Q_after_PET <- residuals(model_PET_Q)
  resid_Q_after_P <- residuals(model_P_Q)
  resid_PET_after_P <- residuals(model_P_PET)
  resid_P_after_PET <- residuals(model_PET_P)
  
  #model_P_Q <- lmodel2(Q ~ P, data=df_annual)
  #model_PET_Q <- lmodel2(Q ~ PET, data=df_annual)
  #model_P_PET <- lmodel2(PET ~ P, data=df_annual)
  #model_PET_P <- lmodel2(P ~ PET, data=df_annual)
  
  #resid_Q_after_PET <- model_PET_Q$regression.residuals[[2]]  
  #resid_Q_after_P <- model_P_Q$regression.residuals[[2]]
  #resid_PET_after_P <- model_P_PET$regression.residuals[[2]]
  #resid_P_after_PET <- model_PET_P$regression.residuals[[2]]
  
  # Partial regression
  partial_P <- lmodel2(resid_Q_after_PET ~ resid_P_after_PET)
  #sens_partial_p <- round(coef(partial_P)[1], 2)
  sens_partial_p <- round(partial_P$regression.results$Slope, 2)
  
  partial_PET <- lmodel2(resid_Q_after_P ~ resid_PET_after_P)
  #sens_partial_pet <- round(coef(partial_PET)[1], 2)
  sens_partial_pet <- round(partial_PET$regression.results$Slope, 2)
  
  cat(paste("Partial Sensitivity to P:", sens_partial_p, "Partial Sensitivity to PET:", sens_partial_pet, "\n"))
  
  # Semi-partial regression
  semi_partial_P <- lm(Q ~ resid_P_after_PET, data = df_annual)
  sens_semi_p <- round(coef(semi_partial_P)[2], 2)
  
  semi_partial_PET <- lm(Q ~ resid_PET_after_P, data = df_annual)
  sens_semi_pet <- round(coef(semi_partial_PET)[2], 2)
  
  cat(paste("Semi-partial Sensitivity to P:", sens_semi_p, "Semi-partial Sensitivity to PET:", sens_semi_pet, "\n"))
  
  # Plotting
  if (plot_results) {
    par(mfrow = c(1,3), mar = c(4,4,2,1))
    
    # Plot relationships between P and PET
    plot(df_annual$P, df_annual$PET,
         col = "grey", pch = 16,
         xlab = "P [mm/yr]", ylab = "PET [mm/yr]")
    abline(model_P_PET, col = "grey")
    slope <- 1 / coef(model_PET_P)[1] # Flip slope (dy/dx becomes dx/dy)
    abline(a = 0, b = slope, col = "grey")
    
    # Plot Q vs. P and Q vs. PET
    xlim_combined <- range(c(df_annual$P, df_annual$PET), na.rm = TRUE)
    ylim_combined <- range(df_annual$Q, na.rm = TRUE)
    plot(df_annual$P, df_annual$Q,
         col = "blue", pch = 16,
         xlab = "P [mm/yr]", ylab = "Q [mm/yr]",
         xlim = xlim_combined, ylim = ylim_combined) # Explicit limits
    points(df_annual$PET, df_annual$Q,
           col = "orange", pch = 16)
    
    # Plot residual relationships
    plot(resid_P_after_PET, resid_Q_after_PET,
         col = "blue", pch = 16,
         xlab = "P residuals [mm/yr]", ylab = "Q residuals [mm/yr]")
    abline(partial_P, col = "blue")
    points(resid_PET_after_P, resid_Q_after_P,
           col = "orange", pch = 16)
    abline(partial_PET, col = "orange")
    
    par(mfrow=c(1,1))
  }
}


# load data
path = "D:/Python/Streamflow_sensitivities/results/camels_DE_timeseries_DE110000.csv"
df_tmp <- read.csv(path)
stepwise_regression_plot(df_tmp$discharge_spec_obs, df_tmp$precipitation_mean, df_tmp$pet_hargreaves, df_tmp$date, plot_results=TRUE, fit_intercept=FALSE)

