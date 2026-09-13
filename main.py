import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# STEP 1: Load the data you made in Part 1
df = pd.read_csv("synthetic_marquis_dataset.csv")

X = df[["H", "S", "V","R","G","B"]]   # the color values (input)
y = df["label"]           # the correct answer for each color (output)

# STEP 2: Split data into "practice" and "test" sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 3: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 4: Check how well it learned
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# STEP 5: Save the trained model to a file
joblib.dump(model, "marquis_classifier.joblib")
print("Saved marquis_classifier.joblib")