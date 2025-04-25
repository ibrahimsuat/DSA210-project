
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy.stats import pearsonr

# ---------- Load and Parse CSV Files ----------
with open("education.csv", "r", encoding="utf-8") as f:
    education_raw = f.read()

with open("traffic.csv", "r", encoding="utf-8") as f:
    traffic_raw = f.read()


# Hypothesis Testing for Correlation between Education and Accidents

r_value, p_value = pearsonr(df["Avg_Education_Years"], df["Accidents_per_1000"])

print("Hypothesis Test:")
print(f"Pearson correlation coefficient (r): {r_value:.3f}")
print(f"p-value: {p_value:.4f}")

alpha = 0.05  # Significance level

if p_value < alpha:
    print("✅ Reject the null hypothesis: There is a statistically significant relationship.")
else:
    print("❌ Fail to reject the null hypothesis: No statistically significant relationship.")