import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("FinanceSpending.csv")

# Encode categorical variable 'Occupation'
le = LabelEncoder()
df['Occupation_encoded'] = le.fit_transform(df['Occupation'])
joblib.dump(le, 'label_encoder.pkl')

# Define all features for training
features = [
    'Income', 'Age', 'Dependents', 'Occupation_encoded', 'City_Tier', 'Rent', 'Loan_Repayment',
    'Insurance', 'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities',
    'Healthcare', 'Education', 'Miscellaneous', 'Desired_Savings_Percentage', 'Disposable_Income'
]

target = [
    'Potential_Savings_Groceries', 'Potential_Savings_Transport', 'Potential_Savings_Eating_Out',
    'Potential_Savings_Entertainment', 'Potential_Savings_Utilities', 'Potential_Savings_Healthcare',
    'Potential_Savings_Education', 'Potential_Savings_Miscellaneous'
]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df[features], 
    df[target], 
    test_size=0.2, 
    random_state=42
)

model_metrics = {}
models = {}

for column in target:
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        objective='reg:squarederror'
    )
    
    y_train_single = y_train[column]
    y_test_single = y_test[column]
    
    model.fit(X_train, y_train_single, eval_set=[(X_test, y_test_single)], verbose=False)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train_single, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test_single, test_pred)),
        'train_r2': r2_score(y_train_single, train_pred),
        'test_r2': r2_score(y_test_single, test_pred),
        'feature_importance': dict(zip(features, model.feature_importances_))
    }
    
    model_metrics[column] = metrics
    models[column] = model
    
    joblib.dump(model, f"savings_prediction_model_{column.lower()}.pkl")

# Save models and metrics
joblib.dump(models, "savings_prediction_models.pkl")
joblib.dump(model_metrics, "model_metrics.pkl")

# Take user input for prediction
income = float(input("Enter Income: "))
occupation = input("Enter Occupation: ")
rent = float(input("Enter Rent: "))
groceries = float(input("Enter Groceries expense: "))
transport = float(input("Enter Transport expense: "))
desired_savings_percentage = float(input("Enter Desired Savings Percentage: "))

# Encode Occupation
occupation_encoded = le.transform([occupation])[0]

# Prepare input data
data_dict = {
    'Income': [income],
    'Occupation_encoded': [occupation_encoded],
    'Rent': [rent],
    'Groceries': [groceries],
    'Transport': [transport],
    'Desired_Savings_Percentage': [desired_savings_percentage]
}
input_data = pd.DataFrame(data_dict)

# Load trained models
models = joblib.load("savings_prediction_models.pkl")

# Make predictions
predictions = {}
for column in target:
    predictions[column] = models[column].predict(input_data)

# Print predictions
predictions_df = pd.DataFrame(predictions)
print("Predicted Savings:")
print(predictions_df)
