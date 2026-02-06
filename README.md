# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2.Data preprocessing: Cleanse data,handle missing values,encode categorical variables.
3.Model Training: Fit logistic regression model on preprocessed data.
4.Model Evaluation: Assess model performance using metrics like accuracyprecisioon,recall.
5.Prediction: Predict placement status for new student data using trained model.
6.End of program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: V.SIRI SAI
RegisterNumber:  212225240181
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/acer/Downloads/Placement_Data (1).csv")
data.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_cols = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

data.head()
X = data.drop(['status', 'salary'], axis=1).values
y = data['status'].values.reshape(-1, 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
m, n = X_train.shape  
w = np.zeros((n, 1))  
b = 0                 

alpha = 0.01          
iterations = 3000     
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
losses = []

for i in range(iterations):
    z = np.dot(X_train, w) + b
    y_hat = sigmoid(z)
    
    dw = (1/m) * np.dot(X_train.T, (y_hat - y_train))
    db = (1/m) * np.sum(y_hat - y_train)
    
    w -= alpha * dw
    b -= alpha * db
    
    loss = -(1/m) * np.sum(
        y_train * np.log(y_hat + 1e-9) + 
        (1 - y_train) * np.log(1 - y_hat + 1e-9)
    )
    losses.append(loss)

print("Training completed")
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs Iterations (Gradient Descent)")
plt.show()
def predict(X):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    return (y_pred >= 0.5).astype(int)
y_pred = predict(X_test)

accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))   
```

## Output:

<img width="832" height="359" alt="image" src="https://github.com/user-attachments/assets/8bd9840b-ffac-4cf6-a46e-4578829fc1e8" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

