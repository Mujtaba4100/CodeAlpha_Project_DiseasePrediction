import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

train_df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

X_train = train_df.drop("prognosis", axis=1)
y_train = train_df["prognosis"]
X_test = test_df.drop("prognosis", axis=1)
y_test = test_df["prognosis"]

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train_encoded)
y_pred_encoded = model.predict(X_test_scaled)
y_pred = le.inverse_transform(y_pred_encoded)

accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print("Test Data Accuracy:", accuracy)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
results.to_csv("prediction.csv", index=False)

joblib.dump(model, "disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(list(X_train.columns), "symptom_list.pkl")
