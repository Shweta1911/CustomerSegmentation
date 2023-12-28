import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load your online retail dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('OnlineRetail.csv')

# Assuming you have a 'label' column indicating the class (e.g., '1' for positive and '0' for negative)
X = df.drop('CustomerID', axis=1)
y = df['Quantity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=50)
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)

# Use Logistic Regression on the reduced features
classifier = LogisticRegression()
classifier.fit(X_train_svd, y_train)

# Make predictions
y_pred = classifier.predict(X_test_svd)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
