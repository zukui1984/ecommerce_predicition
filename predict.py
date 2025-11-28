import pickle
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify

# Load model and encoder
print("Loading model and encoder.")
model = xgb.Booster()
model.load_model('model.json')  # Changed from .bin to .json

with open('dv.pkl', 'rb') as f:
    dv = pickle.load(f)

app = Flask('purchase-prediction')

def prepare_session(session_data):
    """Prepare session features"""
    # Engineer features
    total_pages = (session_data.get('Administrative', 0) + 
                   session_data.get('Informational', 0) + 
                   session_data.get('ProductRelated', 0))
    
    total_duration = (session_data.get('Administrative_Duration', 0) + 
                      session_data.get('Informational_Duration', 0) + 
                      session_data.get('ProductRelated_Duration', 0))
    
    session_data['TotalPages'] = total_pages
    session_data['TotalDuration'] = total_duration
    session_data['AvgTimePerPage'] = total_duration / (total_pages + 1)
    session_data['ProductPageRatio'] = session_data.get('ProductRelated', 0) / (total_pages + 1)
    session_data['IsHighValueVisitor'] = int(session_data.get('PageValues', 0) > 5.0)
    session_data['IsHolidaySeason'] = int(session_data.get('Month', '') in ['Nov', 'Dec'])
    
    # Engagement score
    session_data['EngagementScore'] = (
        (total_pages / 100) * 0.3 +
        (total_duration / 3000) * 0.3 +
        ((1 - session_data.get('BounceRates', 0)) * 0.2) +
        ((1 - session_data.get('ExitRates', 0)) * 0.2)
    )
    
    return session_data

@app.route('/')
def home():
    return jsonify({
        'service': 'E-commerce Purchase Prediction API',
        'version': '1.0',
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict'
        },
        'example': {
            'url': 'POST http://localhost:9696/predict',
            'body': {
                'Administrative': 0,
                'ProductRelated': 5,
                'BounceRates': 0.02,
                'PageValues': 10.5,
                'Month': 'Nov',
                'VisitorType': 'Returning_Visitor'
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    session = request.get_json()
    
    # Prepare features
    session = prepare_session(session)
    
    # Encode
    X = dv.transform([session])
    dmatrix = xgb.DMatrix(X)
    
    # Predict
    purchase_prob = float(model.predict(dmatrix)[0])
    will_purchase = purchase_prob >= 0.5
    
    # Determine action
    if purchase_prob >= 0.7:
        action = "High intent - Show special offer"
    elif purchase_prob >= 0.5:
        action = "Medium intent - Send reminder email"
    elif purchase_prob >= 0.3:
        action = "Low intent - Retargeting ad"
    else:
        action = "Very low intent - No action"
    
    result = {
        'purchase_probability': round(purchase_prob, 4),
        'will_purchase': bool(will_purchase),
        'confidence': 'high' if purchase_prob > 0.7 or purchase_prob < 0.3 else 'medium',
        'recommended_action': action
    }
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'xgboost',
        'version': '1.0'
    })

if __name__ == '__main__':
    print("\nService running on http://0.0.0.0:9696")
    print("  • POST /predict")
    print("  • GET  /health")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=9696)
