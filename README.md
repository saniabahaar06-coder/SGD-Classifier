# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the Iris dataset and separate features (X) and target labels (y).
2.Divide the dataset into training and testing sets using train_test_split().
3.Apply StandardScaler() to normalize the feature values. 
4.Create an SGDClassifier model and fit it using the training data. 5.Predict the test data and calculate accuracy, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: sania bahaar r
RegisterNumber: 25018890
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = SGDClassifier(loss='log_loss', max_iter=1500, random_state=7)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

 
*/
```

## Output:
![prediction of iris species using SGD Classifier](sam.png)
<img width="1919" height="969" alt="Screenshot 2026-03-11 153726" src="https://github.com/user-attachments/assets/24dfddbf-4bf0-4c73-b068-9ad3db103003" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
