import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("datasets/titanic.csv")
print(df.head())
print(df.describe())
print(df.info())
df = df[["Age", "Fare"]]
df = df.dropna()

X = df[["Age"]]
y = df["Fare"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
#Create a linear regression model
model = LinearRegression()
#Train the model (Fare = m * Age + b)
model.fit(X_train, y_train)
print("Slope:",model.coef_[0])
print("Intercept:",model.intercept_)
#Make predictions
y_pred = model.predict(X_test)
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted":y_pred
})
print(comparison.head(5))
mse = mean_squared_error(y_test,y_pred)
r2  = r2_score(y_test,y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

#Visualize the results
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test,y_pred,color='red', label='Predicted')
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Linear Regression: Age vs Fare")
plt.legend()
plt.show()