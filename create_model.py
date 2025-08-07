import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib

# --- 1. Define the training data ---
data = {
    'credit_amount': [2500, 7000, 1500, 8000, 3000, 12000],
    'age': [25, 45, 22, 50, 31, 48],
    'is_homeowner': [0, 1, 0, 1, 1, 1], # 1 for yes, 0 for no
    'risk_class': [1, 1, 1, 0, 1, 0] # 1 is 'good_risk', 0 is 'bad_risk'
}
df = pd.DataFrame(data)
print("Data created successfully!")

# --- 2. Define Features (X) and Target (y) ---
features = ['credit_amount', 'age', 'is_homeowner']
target = 'risk_class'
X = df[features]
y = df[target]

# --- 3. Train the "Black Box" Neural Network Model ---
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
model.fit(X, y)
print("Neural Network model training complete!")

# --- 4. Save the Trained Model to a File ---
filename = 'black_box_loan_model.joblib'
joblib.dump(model, filename)
print(f"✅ Model saved successfully as '{filename}'!")
