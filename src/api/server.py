
from flask import Flask, request, jsonify
import pandas as pd, joblib
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

app = Flask(__name__)
model = joblib.load('results/models/fair_tax_model.joblib')

@app.get('/health')
def health(): return {'status':'ok'}

@app.post('/audit_score')
def audit_score():
    d = request.json
    df = pd.DataFrame([d])
    pred = int(model.predict(df)[0])
    return {'audit_risk': pred}

@app.post('/fairness_check')
def fairness_check():
    payload = request.json
    df = pd.DataFrame(payload['data'])
    y = df['label']
    preds = model.predict(df.drop('label',axis=1))
    dp = demographic_parity_difference(y, preds, sensitive_features=df[payload['feature']])
    eo = equalized_odds_difference(y, preds, sensitive_features=df[payload['feature']])
    return {'demographic_parity': dp, 'equalized_odds': eo}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
