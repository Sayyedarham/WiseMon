import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("FinanceSpending.csv")

le = LabelEncoder()
df['Occupation_encoded'] = le.fit_transform(df['Occupation'])
df['City_Tier_encoded'] = le.fit_transform(df['City_Tier'])

joblib.dump(le, 'label_encoders.pkl')

features = [
    'Income', 'Age', 'Dependents', 'Occupation_encoded', 'City_Tier_encoded',
    'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education',
    'Miscellaneous', 'Desired_Savings_Percentage', 'Disposable_Income'
]

target = [
   'Potential_Savings_Groceries','Potential_Savings_Transport','Potential_Savings_Eating_Out','Potential_Savings_Entertainment',
    'Potential_Savings_Utilities','Potential_Savings_Healthcare','Potential_Savings_Education','Potential_Savings_Miscellaneous'
]

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
    
    model.fit(
        X_train,
        y_train_single,
        eval_set=[(X_test, y_test_single)],
        verbose=False
    )
    
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

joblib.dump(models, "savings_prediction_models.pkl")
joblib.dump(model_metrics, "model_metrics.pkl")

print("\nModel Performance Summary:")
print("=" * 50)
for target_var, metrics in model_metrics.items():
    print(f"\nTarget Variable: {target_var}")
    print(f"Train RMSE: {metrics['train_rmse']:.2f}")
    print(f"Test RMSE: {metrics['test_rmse']:.2f}")
    print(f"Train R²: {metrics['train_r2']:.3f}")
    print(f"Test R²: {metrics['test_r2']:.3f}")
    
    print("\nTop 5 Important Features:")
    sorted_features = dict(sorted(metrics['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:5])
    for feature, importance in sorted_features.items():
        print(f"{feature}: {importance:.3f}")
    print("-" * 50)
