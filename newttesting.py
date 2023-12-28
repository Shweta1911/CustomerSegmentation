# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st

# Load the data
df = pd.read_excel('Online Retail.xlsx')

# Data preprocessing...
# (Add any necessary preprocessing steps here)

# Create a binary classification target variable (e.g., liked or not liked)
df['Liked'] = (df['Quantity'] > 0).astype(int)

# Create features and target variable
X = df[['CustomerID', 'StockCode']]
y = df['Liked']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier (you can choose a different model)
model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy}")

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(conf_matrix)

# Display feature importances
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_})
st.write("Feature Importances:")
st.write(feature_importances)
