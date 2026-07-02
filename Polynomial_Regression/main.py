import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("datasets/Position_Salaries.csv")
print(df.head())
print(df.info())

X = df[["Level"]]
y =df[["Salary"]]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 42)
#Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

#Make predictions
linear_pred = linear_model.predict(X_test)
poly_pred = poly_model.predict(X_test_poly)
#Evaluate the models
print("Linear Regression:")
print("MSE:",mean_squared_error(y_test, linear_pred))
print("R-squared:", r2_score(y_test, linear_pred))
print("\nPolynomial Regression:")
print("MSE:",mean_squared_error(y_test, poly_pred))
print("R-squared:", r2_score(y_test, poly_pred))

X_plot = X.sort_values(by="Level")
X_plot_poly = poly.transform(X_plot)
y_plot = poly_model.predict(X_plot_poly)
#Visualize the results
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_plot, y_plot, color='red', label='Polynomial Regression')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression: Level vs Salary")
plt.legend()
plt.show()
