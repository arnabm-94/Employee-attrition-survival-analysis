import pandas as pd
import joblib

# 1. Load model
model = joblib.load("cox_model.pkl")

# 2. Load test data
df = pd.read_csv("test_data.csv")

# 3. Print Columns
print("Columns in CSV:")
print(df.columns.tolist())


# 4. Predict survival function (probability of staying)
survival = model.predict_survival_function(df)

# 5. Display results
for i, employee in enumerate(survival):
    print(f"\nEmployee {i+1} survival probabilities:")
    print(employee)
