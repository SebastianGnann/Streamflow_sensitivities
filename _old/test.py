import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
from scipy import stats
from mpl_toolkits.basemap import Basemap
import time
import functions.util_TurcPike as util_TurcPike

#mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

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
df = pd.read_csv(results_path + 'camels_DE_sensitivities.csv')

p_lims = [-0.2, 2.8]
pet_lims = [-2.2, 1.6]
# plot data
# compare sens_P to sens_P_alt1 etc.
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7,3),constrained_layout=True)
im1=ax1.scatter(df["sens_P_alt1"],df["sens_P"],s=10,c="tab:blue",alpha=0.8,lw=0,label="dQ/dP")
ax1.plot(p_lims, p_lims, color='grey', linestyle='--', linewidth=1)
ax1.set_xlabel("dQ/dP intercept=True (2)[-]")
ax1.set_ylabel("dQ/dP base [-]")
ax1.set_xlim(p_lims)
ax1.set_ylim(p_lims)
ax1.grid()
ax1.legend()
im2=ax2.scatter(df["sens_PET_alt1"],df["sens_PET"],s=10,c="tab:orange",alpha=0.8,lw=0,label="dQ/dPET")
ax2.plot(pet_lims, pet_lims, color='grey', linestyle='--', linewidth=1)
ax2.set_xlabel("dQ/dPET intercept=True (2) [-]")
ax2.set_ylabel("dQ/dPET base [-]")
ax2.set_xlim(pet_lims)
ax2.set_ylim(pet_lims)
ax2.grid()
ax2.legend()
plt.show()

# compare sens_P to sens_P_alt1 etc.
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7,3),constrained_layout=True)
im1=ax1.scatter(df["sens_P_alt1"],df["sens_P_alt2"],s=10,c="tab:blue",alpha=0.8,lw=0,label="dQ/dP")
ax1.plot(p_lims, p_lims, color='grey', linestyle='--', linewidth=1)
ax1.set_xlabel("dQ/dP intercept=True (2) [-]")
ax1.set_ylabel("dQ/dP delta=True (3) [-]")
ax1.set_xlim(p_lims)
ax1.set_ylim(p_lims)
ax1.grid()
ax1.legend()
im2=ax2.scatter(df["sens_PET_alt1"],df["sens_PET_alt2"],s=10,c="tab:orange",alpha=0.8,lw=0,label="dQ/dPET")
ax2.plot(pet_lims, pet_lims, color='grey', linestyle='--', linewidth=1)
ax2.set_xlabel("dQ/dPET intercept=True (2) [-]")
ax2.set_ylabel("dQ/dPET delta=True (3) [-]")
ax2.set_xlim(pet_lims)
ax2.set_ylim(pet_lims)
ax2.grid()
ax2.legend()
plt.show()

# compare sens_P to sens_P_alt1 etc.
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7,3),constrained_layout=True)
im1=ax1.scatter(df["sens_P_Budyko"],df["sens_P"],s=10,c="tab:blue",alpha=0.5,lw=0,label="1")
im1=ax1.scatter(df["sens_P_Budyko"],df["sens_P_alt2"],s=10,c="tab:purple",alpha=0.5,lw=0,label="2")
ax1.plot(p_lims, p_lims, color='grey', linestyle='--', linewidth=1)
ax1.set_xlabel("dQ/dP Budyko [-]")
ax1.set_ylabel("dQ/dP options [-]")
ax1.set_xlim(p_lims)
ax1.set_ylim(p_lims)
ax1.grid()
ax1.legend()
im2=ax2.scatter(df["sens_PET_Budyko"],df["sens_PET"],s=10,c="tab:orange",alpha=0.5,lw=0,label="1")
im2=ax2.scatter(df["sens_PET_Budyko"],df["sens_PET_alt2"],s=10,c="tab:red",alpha=0.5,lw=0,label="2")
ax2.plot(pet_lims, pet_lims, color='grey', linestyle='--', linewidth=1)
ax2.set_xlabel("dQ/dPET Budyko [-]")
ax2.set_ylabel("dQ/dPET options [-]")
ax2.set_xlim(pet_lims)
ax2.set_ylim(pet_lims)
ax2.grid()
ax2.legend()
plt.show()
