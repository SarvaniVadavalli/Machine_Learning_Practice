import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("datasets/titanic.csv")
#Explore datasets
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
print(df.columns)
#Check missing values
print(df.isnull().sum())
#Handle missing values
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
print(df.isnull().sum())
#Remove unnecessary columns
df=df.drop(columns=(["Cabin","Ticket","Name"]))
print(df.columns)

#Check for duplicates
print(df.duplicated().sum())
df = df.drop_duplicates()
#Encode categorical variables
df["Sex"] = df["Sex"].map({
    "male":0,
    "female":1
})
df["Embarked"]=df["Embarked"].map({
    "S":0,
    "C":1,
    "Q":2
})
X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
y = df["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
print(X_scaled.head())
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
comparisom = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(comparisom.head())

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)
print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report")
print(classification_report(y_test,y_pred))

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.show()