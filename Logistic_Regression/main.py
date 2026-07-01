import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("datasets/titanic.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df = df[["Pclass", "Sex", "Age", "Fare", "Survived","SibSp", "Parch"]]

df = df.dropna()

df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})
print(df.head())
X = df[["Pclass", "Sex", "Age", "Fare","SibSp","Parch"]]
y = df["Survived"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print(comparison.head(5))
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")
cm = confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:")
print(cm)
print(classification_report(y_test,y_pred))

