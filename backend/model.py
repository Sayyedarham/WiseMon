import pandas as pd
import numpy as np
import joblib
import sys
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# File paths
LABEL_ENCODER_FILE = "label_encoders.pkl"
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
    'City_Tier_encoded': 1,  # assuming tier 1 is default
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

def train_model():
    """Train XGBoost models for each target variable and save them."""
    df = pd.read_csv("FinanceSpending.csv")

    # Label encoding categorical features
    le = LabelEncoder()
    df['Occupation_encoded'] = le.fit_transform(df['Occupation'])
    df['City_Tier_encoded'] = le.fit_transform(df['City_Tier'])
    joblib.dump(le, LABEL_ENCODER_FILE)

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
    print("Training complete. Models saved.")

def prepare_feature_vector(income, occupation, rent, groceries, transport, desired_savings_percentage):
    """Prepare the feature vector for inference by filling missing values."""
    le = joblib.load(LABEL_ENCODER_FILE)
    occupation_encoded = le.transform([occupation])[0]

    feature_vector = {
        'Income': income,
        'Age': default_values['Age'],
        'Dependents': default_values['Dependents'],
        'Occupation_encoded': occupation_encoded,
        'City_Tier_encoded': default_values['City_Tier_encoded'],
        'Rent': rent,
        'Loan_Repayment': default_values['Loan_Repayment'],
        'Insurance': default_values['Insurance'],
        'Groceries': groceries,
        'Transport': transport,
        'Eating_Out': default_values['Eating_Out'],
        'Entertainment': default_values['Entertainment'],
        'Utilities': default_values['Utilities'],
        'Healthcare': default_values['Healthcare'],
        'Education': default_values['Education'],
        'Miscellaneous': default_values['Miscellaneous'],
        'Desired_Savings_Percentage': desired_savings_percentage,
        'Disposable_Income': default_values['Disposable_Income']
    }

    return pd.DataFrame([feature_vector])

def predict_savings(income, occupation, rent, groceries, transport, desired_savings_percentage):
    """Make savings predictions using trained models."""
    models = joblib.load(MODELS_FILE)
    X_input = prepare_feature_vector(income, occupation, rent, groceries, transport, desired_savings_percentage)

    predictions = {}
    for column, model in models.items():
        predictions[column] = model.predict(X_input)[0]

    return predictions

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
    

        results = predict_savings(**user_inputs)
        print("Predicted Potential Savings:", results)
