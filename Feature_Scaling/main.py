from pyexpat import features

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,  MinMaxScaler

df = pd.read_csv("datasets/titanic.csv")
df = df[["Age", "Fare", "SibSp", "Parch"]]
df = df.dropna()

print("Original data\n")
print(df.head())
#Standardization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
standardized_df = pd.DataFrame(df_scaled, columns=df.columns)
print("\nStandardized data\n")
print(standardized_df.head())

#Min-Max Scaling
min_max_scaler = MinMaxScaler()
df_min_max_scaler = min_max_scaler.fit_transform(df)
min_max_scaler_df = pd.DataFrame(df_min_max_scaler,columns = df.columns)
print("\nMin-Max Scaled data\n")
print(min_max_scaler_df.head())

print("\nOriginal data description\n")
print(df.describe())
print("\nStandardized data description\n")
print(standardized_df.describe())
print("\nMin-Max Scaled data description\n")
print(min_max_scaler_df.describe())

features = ["Age", "Fare", "SibSp", "Parch"]
for feature in features:
    plt.figure(figsize=(12,4))
    #Original data
    plt.subplot(1,3,1)
    plt.hist(df[feature],bins=20)
    plt.title(f"Original {feature} distribution")
    #Standardized data
    plt.subplot(1,3,2)
    plt.hist(standardized_df[feature],bins=20)
    plt.title(f"Standardized {feature} distribution")
    #Min-Max Scaled data
    plt.subplot(1,3,3)
    plt.hist(min_max_scaler_df[feature],bins=20)
    plt.title(f"Min-Max Scaled {feature} distribution")
    plt.show()