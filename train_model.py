import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

data = pd.read_csv("dataset.csv")

print("\nDataset loaded successfully!")
print("Total records:", len(data))


# ---------------------------------------
# 2. Encode Protocol
# ---------------------------------------

protocol_encoder = LabelEncoder()

data["protocol"] = protocol_encoder.fit_transform(
    data["protocol"]
)


# ---------------------------------------
# 3. Encode Target
# ---------------------------------------

label_encoder = LabelEncoder()

data["label"] = label_encoder.fit_transform(
    data["label"]
)


# ---------------------------------------
# 4. Features and Target
# ---------------------------------------

features = [
    "packet_size",
    "connection_duration",
    "failed_logins",
    "requests_per_second",
    "port_number",
    "protocol"
]

X = data[features]
y = data["label"]


# ---------------------------------------
# 5. Split Dataset
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ---------------------------------------
# 6. Create Random Forest Model
# ---------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ---------------------------------------
# 7. Train Model
# ---------------------------------------

model.fit(X_train, y_train)

print("\nAI model training completed successfully!")


# ---------------------------------------
# 8. Predictions
# ---------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------
# 9. Accuracy
# ---------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ---------------------------------------
# 10. Classification Report
# ---------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ---------------------------------------
# 11. Confusion Matrix
# ---------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ---------------------------------------
# 12. Save Everything
# ---------------------------------------

model_data = {
    "model": model,
    "protocol_encoder": protocol_encoder,
    "label_encoder": label_encoder,
    "accuracy": accuracy,
    "confusion_matrix": cm,
    "features": features
}


with open("model.pkl", "wb") as file:

    pickle.dump(
        model_data,
        file
    )


print("\nModel saved successfully!")

print("\nTraining completed!")