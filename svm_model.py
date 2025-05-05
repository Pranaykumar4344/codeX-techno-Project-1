import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset correctly (assuming Iris.csv has a header row)
data = pd.read_csv('Iris.csv')  # No need for 'header=None'

# Check actual column names
print("Column names:", data.columns)

# Rename columns to consistent names (optional but cleaner)
data.rename(columns={
    'SepalLengthCm': 'sepal_length',
    'SepalWidthCm': 'sepal_width',
    'PetalLengthCm': 'petal_length',
    'PetalWidthCm': 'petal_width',
    'Species': 'species'
}, inplace=True)

# Display class distribution
print("\nClass distribution:")
print(data['species'].value_counts())

# Encode target
le = LabelEncoder()
y = le.fit_transform(data['species'])

# Features
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the SVM model: {accuracy * 100:.2f}%")

# Save the model and label encoder
joblib.dump(model, 'svm_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
