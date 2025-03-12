import pandas as pd
import numpy as np
import joblib
import pickle
import sys
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# File paths
MODEL_METRICS_FILE = "model_metrics.pkl"
MODELS_FILE = "savings_prediction_models.pkl"

# Feature and target columns
features = [
    'Income', 'Age', 'Dependents', 'Occupation_encoded', 'City_Tier_encoded',
    'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education',
    'Miscellaneous', 'Desired_Savings_Percentage', 'Disposable_Income'
]

target = [
    'Potential_Savings_Groceries', 'Potential_Savings_Transport',
    'Potential_Savings_Eating_Out', 'Potential_Savings_Entertainment',
    'Potential_Savings_Utilities', 'Potential_Savings_Healthcare',
    'Potential_Savings_Education', 'Potential_Savings_Miscellaneous'
]

# Default values for missing features in inference mode
default_values = {
    'Age': 35,
    'Dependents': 1,
    'City_Tier_encoded': 1,
    'Loan_Repayment': 500,
    'Insurance': 100,
    'Eating_Out': 150,
    'Entertainment': 100,
    'Utilities': 200,
    'Healthcare': 100,
    'Education': 150,
    'Miscellaneous': 50,
    'Disposable_Income': 1000
}

df = pd.read_csv("FinanceSpending.csv")

# Label encoding categorical features
le = LabelEncoder()
df['Occupation_encoded'] = le.fit_transform(df['Occupation'])
df['City_Tier_encoded'] = le.fit_transform(df['City_Tier'])

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

models = {}
model_metrics = {}

for column in target:
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        objective='reg:squarederror'
    )

    model.fit(
        X_train, y_train[column],
        eval_set=[(X_test, y_test[column])],
        verbose=False
    )

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train[column], train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test[column], test_pred)),
        'train_r2': r2_score(y_train[column], train_pred),
        'test_r2': r2_score(y_test[column], test_pred),
        'feature_importance': dict(zip(features, model.feature_importances_))
    }

    model_metrics[column] = metrics
    models[column] = model
    joblib.dump(model, f"savings_prediction_model_{column.lower()}.pkl")

# Save all models and metrics
joblib.dump(models, MODELS_FILE)
joblib.dump(model_metrics, MODEL_METRICS_FILE)
with open('XGBR.pkl', 'wb') as file:
    pickle.dump(models, file)

print("Training complete. Models saved and ready for upload.")
