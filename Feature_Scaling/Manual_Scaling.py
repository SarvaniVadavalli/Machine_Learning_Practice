import pandas as pd

df = pd.read_csv("datasets/titanic.csv")
df = df[["Age", "Fare"]]
df = df.dropna()
#Age Standardization
age_mean= df["Age"].mean()
age_std = df["Age"].std()
df["Age Standardized"] = (df["Age"] - age_mean) / age_std
#Fare Standardization
fare_mean = df["Fare"].mean()
fare_std = df["Fare"].std()
df["Fare Standardized"] = (df["Fare"] - fare_mean) / fare_std
#Min-Max Scaling
age_min = df["Age"].min()
age_max = df["Age"].max()
df["Age Standardized"]=(df["Age"] - age_min)/(age_max - age_min)
fare_min = df["Fare"].min()
fare_max = df["Fare"].max()
df["Fare Standardized"]=(df["Fare"] - fare_min)/(fare_max - fare_min)

print(df.head())
print("\nStatistics:\n")
print(df.describe())