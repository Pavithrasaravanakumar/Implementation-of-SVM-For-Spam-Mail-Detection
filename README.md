# Implementation-of-SVM-For-Spam-Mail-Detection
## NAME:PAVITHRA S
## REG NO:212223220073

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries.

2.Read the CSV file and display data using head().

3.Split the dataset using train_test_split().

4.Calculate predictions and accuracy.

5.Print the outputs.

6.End the program.

## Program:
```
import chardet, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

# Detect encoding
with open('spam.csv', 'rb') as f:
    print(chardet.detect(f.read(100000)))

```

## Output:
![image](https://github.com/user-attachments/assets/97cbc613-e2e1-4129-896f-7a277ececa23)

```
# Load data
data = pd.read_csv('spam.csv', encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())
```
## Output:
![image](https://github.com/user-attachments/assets/3e0792a5-71df-43cc-ab3a-a8c09e1fd055)

```
# Split data
x = data['v1'].values   # Labels
y = data['v2'].values   # Messages
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Text vectorization
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# Train & predict
model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predictions:", y_pred)
```
## Output:

![image](https://github.com/user-attachments/assets/97e0a6c5-82f4-4565-8371-0fcb3151904f)

```
# Train & predict
model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predictions:", y_pred)
```
## Output:
![image](https://github.com/user-attachments/assets/8dec9d02-faa7-4f57-83de-fda2ed02b4b9)
```
# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
## Output:
![image](https://github.com/user-attachments/assets/7239ac81-470e-4f67-bfc5-8ec3f9d71cc3)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
