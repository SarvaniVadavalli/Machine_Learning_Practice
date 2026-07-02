import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("datasets/titanic.csv")
df = df[["Pclass", "Sex", "Age", "SibSp", "Parch","Fare"]]
df["Sex"] = df["Sex"].map({"male":0, "female":1})

df = df.dropna()
print(df.isnull().sum())

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch"]]
y = df["Fare"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept:", model.intercept_)
coefficients=pd.DataFrame({"Feature":X.columns,
                           "Coefficient":model.coef_})
print(coefficients)
y_pred= model.predict(X_test)
comparison = pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
print(comparison.head())

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare")
plt.show()