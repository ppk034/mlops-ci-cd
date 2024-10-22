# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset from local CSV file
data = pd.read_csv('winequality-red.csv', delimiter=';')

# Define feature columns and target variable (quality)
X = data.drop('quality', axis=1)  # Features (all columns except 'quality')
y = data['quality']  # Target (wine quality)

# Simplify the problem for classification: Classify quality as good (1) or not (0)
y = y.apply(lambda q: 1 if q >= 7 else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
