import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset path
dataset_path = r"C:\Users\hp\OneDrive\Desktop\HousePricePrediction\archive (3)\train\train"

data = []
labels = []

IMG_SIZE = 64

cat_count = 0
dog_count = 0

print("Loading images...")

for img_name in os.listdir(dataset_path):

    # Take only 1000 cat images
    if img_name.startswith("cat") and cat_count < 1000:
        label = 0
        cat_count += 1

    # Take only 1000 dog images
    elif img_name.startswith("dog") and dog_count < 1000:
        label = 1
        dog_count += 1

    else:
        continue

    try:
        img_path = os.path.join(dataset_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        data.append(img.flatten())
        labels.append(label)

    except:
        pass

print("Cats loaded:", cat_count)
print("Dogs loaded:", dog_count)

X = np.array(data)
y = np.array(labels)

print("Dataset shape:", X.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training SVM model...")

# Create SVM
model = SVC(kernel='linear')

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))