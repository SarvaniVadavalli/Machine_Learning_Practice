import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.metrics import (
               accuracy_score,
               confusion_matrix,
               classification_report,
               ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/titanic.csv")
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df = df.drop(columns = [
          "PassengerId",
          "Name",
          "Ticket",
          "Cabin"
])
df=df.drop_duplicates()
df["Sex"] = df["Sex"].map({
    "male":0,
    "female":1
})
df["Embarked"] = df["Embarked"].map({

    "S":0,
    "C":1,
    "Q":2
})
X = df[[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]]

y = df["Survived"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
comparision = pd.DataFrame({
    "Actual":y_test,
    "Predicted":y_test
})
print(comparision.head())
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(
    y_test,y_pred
))

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    cmap="Blues"
)
plt.title("KNN Confusion Matrix")
plt.show()
