import numpy as np
import matplotlib.pyplot as plt

def dQ_dEP(P, EP, n=2.5):
    return -((1 + (EP / P)**n)**((-n+1)/n))

def dQ_dP(P, EP, n=2.5):
    return 1 - ((1 + (P / EP)**n)**((-n+1)/n))

# Generate a range of aridity indices (defined as P/EP)
P_over_EP = np.linspace(0.2, 5, 1000)  # Aridity index from 0.2 to 5
EP = 1  # Assume EP = 1 for simplicity
P = P_over_EP * EP

# Calculate streamflow sensitivities
dQdEP = dQ_dEP(P, EP)
dQdP = dQ_dP(P, EP)

# Plotting both relationships in a single plot
plt.figure(figsize=(10, 6))

plt.plot(P_over_EP, dQdP, label='dQ/dP', color='tab:blue', linewidth=2)
plt.plot(P_over_EP, dQdEP, label='dQ/dEP', color='tab:orange', linewidth=2)

plt.xlabel('Aridity Index (P/PET)', fontsize=14)
plt.ylabel('Streamflow Sensitivity', fontsize=14)
plt.title('Streamflow Sensitivity to P and PET vs Aridity Index', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# Add a vertical line at P/EP = 1
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.5, label='P/PET = 1')

plt.xlim(0.2, 5)
plt.ylim(-1, 1)

plt.tight_layout()
plt.show()
