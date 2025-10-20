
import pandas as pd, numpy as np, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

def main():
    df = pd.read_csv('data/synthetic_taxpayers.csv')
    X = df.drop('audit_risk_label', axis=1)
    y = df['audit_risk_label']

    cat_cols = ['gender','ethnicity','filing_status']
    preproc = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)], remainder='passthrough')
    model = Pipeline([('prep', preproc), ('clf', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    print(f"AUC={auc:.3f}, ACC={acc:.3f}")
    os.makedirs('results/models', exist_ok=True)
    joblib.dump(model, 'results/models/fair_tax_model.joblib')

if __name__ == '__main__':
    main()
