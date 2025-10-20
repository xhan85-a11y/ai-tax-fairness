
import pandas as pd, joblib, json, os
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference

def main():
    df = pd.read_csv('data/synthetic_taxpayers.csv')
    model = joblib.load('results/models/fair_tax_model.joblib')
    X = df.drop('audit_risk_label', axis=1)
    y = df['audit_risk_label']
    preds = model.predict(X)

    fairness_report = {
        "gender_parity_diff": demographic_parity_difference(y, preds, sensitive_features=df['gender']),
        "ethnicity_parity_diff": demographic_parity_difference(y, preds, sensitive_features=df['ethnicity']),
        "equalized_odds_diff": equalized_odds_difference(y, preds, sensitive_features=df['gender']),
    }

    os.makedirs('results/reports', exist_ok=True)
    with open('results/reports/fairness_report.json','w') as f:
        json.dump(fairness_report,f,indent=2)
    print('Fairness report saved.')

if __name__ == '__main__':
    main()
