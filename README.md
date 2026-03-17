# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and clean the dataset, converting labels to numeric form.

2.Convert text messages into numerical features using TF-IDF.

3.Train the SVM model on the training data.

4.Predict and evaluate accuracy, then classify new messages as spam or ham.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SAKTHI SABARISH P
RegisterNumber: 212225040360 
*/
```
```

# Import necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data

df = pd.read_csv("C:/Users/acer/Downloads/spam.csv", encoding='latin-1')

# Keep only the first two columns and rename them

df = df.iloc[:, :2]
df.columns = ['label', 'message']

# Convert labels(ham = 0, spam = 1) :

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Remove any rows with missing values

df = df.dropna()

# Display basic info

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['label'].value_counts())

# Split data into features (X) and target (y)

X = df['message']
y = df['label']

# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train SVM model

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vectorized, y_train)

# Make predictions

y_pred = svm_model.predict(X_test_vectorized)

# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Test with some example messages

test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "CONGRATULATIONS! You've won a FREE cruise to the Andaman! Call now to claim your prize!",
    "Can you pick up some milk on your way home?",
    "URGENT! Your account has been suspended. Click here to verify your details immediately."
]

print("\n" + "="*50)
print("TESTING WITH EXAMPLE MESSAGES:")
print("="*50)

for msg in test_messages:
    msg_vectorized = vectorizer.transform([msg])
    prediction = svm_model.predict(msg_vectorized)[0]
    result = "SPAM" if prediction == 1 else "HAM"
    print(f"Message: {msg[:50]}... -> {result}")

```

## Output:
![Screenshot_17-3-2026_201457_localhost](https://github.com/user-attachments/assets/0d07c0c9-cafe-47f4-bff1-2f3bf1bd8952)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
