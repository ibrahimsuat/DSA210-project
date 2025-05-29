
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- Load and Parse CSV Files ----------
with open("education.csv", "r", encoding="utf-8") as f:
    education_raw = f.read()

with open("traffic.csv", "r", encoding="utf-8") as f:
    traffic_raw = f.read()

# ---------- Extract Province Names ----------
province_line = re.search(r"Satırlar\|\|\|(.+?)\|?\n", education_raw).group(1)
province_parts = province_line.split("|")
provinces = [p.rsplit("-", 1)[0] for p in province_parts]

# ---------- Extract Education Data ----------
edu_line = re.search(r"Ortalama Eğitim Süresi.*?\|([\d\.\|]+)", education_raw)
education_values = [float(x) for x in edu_line.group(1).split("|") if x.strip()]
education_values = [min(x, 12.0) for x in education_values]  # Clamp to max 12 years because there is one city with 2000 years of education

# ---------- Extract Traffic Data ----------
traffic_line = re.search(r"Ölümlü Yaralanmalı Trafik Kaza Sayısı.*?\|([\d\.\|]+)", traffic_raw)
traffic_values = [float(x) for x in traffic_line.group(1).split("|") if x.strip()]

# ---------- Population estimates (I CANNOT GET PYTHON TO READ .XLS FILES WHICH TUIK PROVIDES) ----------
population_dict = {
    'Adana': 2270925, 'Adıyaman': 635169, 'Afyonkarahisar': 747555, 'Aksaray': 429977, 'Amasya': 335331,
    'Ankara': 5663322, 'Antalya': 2697221, 'Ardahan': 96872, 'Artvin': 169501, 'Aydın': 1134036,
    'Ağrı': 524644, 'Balıkesir': 1290035, 'Bartın': 203351, 'Batman': 634491, 'Bayburt': 86021,
    'Bilecik': 372194, 'Bingöl': 281205, 'Bitlis': 352674, 'Bolu': 320865, 'Burdur': 273716,
    'Bursa': 3472468, 'Denizli': 1050000, 'Diyarbakır': 1822163, 'Düzce': 405131, 'Edirne': 414714,
    'Elazığ': 587802, 'Erzincan': 239223, 'Erzurum': 756893, 'Eskişehir': 898369, 'Gaziantep': 2172022,
    'Giresun': 419555, 'Gümüşhane': 141702, 'Hakkari': 280514, 'Hatay': 1692043, 'Isparta': 453994,
    'Iğdır': 209540, 'Kahramanmaraş': 1181543, 'Karabük': 252058, 'Karaman': 263058, 'Kars': 292660,
    'Kastamonu': 383373, 'Kayseri': 1450906, 'Kilis': 155179, 'Kocaeli': 2062740, 'Konya': 2326610,
    'Kütahya': 571554, 'Kırklareli': 369433, 'Kırıkkale': 280379, 'Kırşehir': 244519, 'Malatya': 812580,
    'Manisa': 1431380, 'Mardin': 868716, 'Mersin': 1918889, 'Muğla': 848313, 'Muş': 303010,
    'Nevşehir': 310011, 'Niğde': 772872, 'Ordu': 559405, 'Osmaniye': 344016, 'Rize': 1130602,
    'Sakarya': 1390420, 'Samsun': 1355552, 'Siirt': 331311, 'Sinop': 218408, 'Sivas': 637723,
    'Tekirdağ': 1130402, 'Tokat': 596454, 'Trabzon': 832178, 'Tunceli': 86157, 'Uşak': 370509,
    'Van': 591204, 'Yalova': 599698, 'Yozgat': 841556, 'Zonguldak': 975763, 'Çanakkale': 273031,
    'Çankırı': 455177, 'Çorum': 597834, 'İstanbul': 15704185, 'İzmir': 4320519, 'Şanlıurfa': 2330053,
    'Şırnak': 557605
}

# ---------- Create DataFrame ----------
df = pd.DataFrame({
    "Province": provinces[:len(education_values)],
    "Avg_Education_Years": education_values[:len(provinces)],
    "Traffic_Accidents": traffic_values[:len(provinces)]
})

# Remove Istanbul
df = df[df["Province"] != "İstanbul"]

# Match population
df["Population"] = df["Province"].map(population_dict)

# Drop missing
df = df.dropna()

# Calculate per capita
df["Accidents_per_1000"] = df["Traffic_Accidents"] / (df["Population"] / 1000)

# ---------- Correlation ----------
r, p = pearsonr(df["Avg_Education_Years"], df["Accidents_per_1000"])

# ---------- Plot ----------
plt.figure(figsize=(14, 8))
sns.regplot(
    data=df,
    x="Accidents_per_1000",
    y="Avg_Education_Years",
    scatter_kws={"color": "teal", "s": 60, "alpha": 0.7},
    line_kws={"color": "orangered", "lw": 2},
    logx=True
)

plt.xscale("log") #for log scale
plt.gca().xaxis.set_major_formatter(ScalarFormatter()) # to show numbers in plain format
plt.ticklabel_format(style='plain', axis='x')

plt.title("Traffic Accidents per 1000 People vs. Avg. Years of Schooling (2023, Turkey)")
plt.xlabel("Traffic Accidents per 1000 People (Log Scale)")
plt.ylabel("Average Years of Schooling")
plt.tight_layout()

# Annotate almost all
for _, row in df.iterrows():
    plt.text(row["Accidents_per_1000"] * 1.01, row["Avg_Education_Years"] + 0.05,
             row["Province"], fontsize=7, alpha=0.8)

# Add correlation text
plt.text(
    0.02, 11.5,
    f"p = {p:.4f}",
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
)

plt.show()

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

# ---------- Machine Learning Model ----------
# Define features and target
X = df[["Avg_Education_Years"]]
y = df["Accidents_per_1000"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n--- ML Evaluation ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

X = df[["Avg_Education_Years", "Accidents_per_1000"]]
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_transform(X_scaled)
