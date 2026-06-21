import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
train_data = pd.read_csv(
    r"C:\Users\hp\OneDrive\Desktop\HousePricePrediction\archive (3)\sign_mnist_train.csv"
)

test_data = pd.read_csv(
    r"C:\Users\hp\OneDrive\Desktop\HousePricePrediction\archive (3)\sign_mnist_test.csv"
)

# Features and labels
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']

X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

print("Training SVM model...")

# Create model
model = SVC(kernel='linear')

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))