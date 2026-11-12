
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

columns = [
    "buying", "maint", "doors",
    "persons", "lug_boot", "safety", "class"
]

df = pd.read_csv(url, header=None, names=columns)

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Step 2: Separate input and target variables
X = df.drop("class", axis=1)
y = df["class"]

# Step 3: Encode categorical input features
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 5: Create and train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Step 6: Predict the test data
y_pred = model.predict(X_test)

print("\nPredicted classes:")
print(label_encoder.inverse_transform(y_pred[:10]))

print("\nActual classes:")
print(label_encoder.inverse_transform(y_test[:10]))

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Step 8: Display classification report
print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))

# Step 9: Plot confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=range(len(label_encoder.classes_))
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# Step 10: Predict the class of a new car
new_car = pd.DataFrame([{
    "buying": "vhigh",
    "maint": "high",
    "doors": "4",
    "persons": "more",
    "lug_boot": "big",
    "safety": "high"
}])

new_car_encoded = encoder.transform(new_car)

prediction = model.predict(new_car_encoded)

predicted_class = label_encoder.inverse_transform(prediction)

print("\nNew car evaluation:", predicted_class[0])

print("\nRandom Forest Classification Completed Successfully!")
