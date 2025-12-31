# Capstone Project - Online Shoppers Purchasing Intention  

## Problem

Only **15.5%** of visitors make a purchase. This model predicts who will buy so e-commerce sites can:
- Show targeted offers to high-intent visitors
- Send cart reminders to medium-intent users
- Optimize ad spend on likely buyers

**Impact:** 100k visitors → $200k/month extra revenue targeting top 20%

---

## Dataset

**[UCI ML Repository - Online Shoppers Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)**

- 12,330 sessions from 1 year
- 18 features: page views, durations, bounce rates, traffic sources
- Target: Purchase (15.5%) vs No Purchase (84.5%)

![Data Distribution](screenshots/03-EDA.jpg)

---

## Quick Start

### Install
```
git clone https://github.com/zukui1984/ecommerce_predicition.git
cd ecommerce_predicition

pip install pipenv
pipenv install
```
### Train
```
pipenv run python train.py
```

![Training](screenshots/01-training-results.jpg)

### Run Service
```
### Local
pipenv run python predict.py

### Docker
docker build -t purchase-prediction .
docker run -p 9696:9696 purchase-prediction
```

![Service](screenshots/02-service-running.jpg)

### Predict
```
curl -X POST http://localhost:9696/predict
-H "Content-Type: application/json"
-d '{
"Administrative": 0,
"ProductRelated": 8,
"ProductRelated_Duration": 450,
"BounceRates": 0.02,
"ExitRates": 0.03,
"PageValues": 15.5,
"Month": "Nov",
"VisitorType": "Returning_Visitor",
"Weekend": false
}'
```

**Response:**
```
{
"purchase_probability": 0.7823,
"will_purchase": true,
"recommended_action": "High intent - Show special offer"
}
```

---

## Results

| Metric | Value |
|--------|-------|
| **AUC** | **0.9164** |
| Accuracy | 89.2% |
| Precision | 78% |
| Recall | 64% |

### Models Tested

![Models](screenshots/06-model-comparison.jpg)

- Random Forest: 0.9325 AUC
- **XGBoost: 0.9227 AUC** ⭐ (chosen)
- Decision Tree: 0.9208 AUC
- Logistic Regression: 0.9172 AUC

---

## Key Insights

### Conversion by Month & Visitor Type

![Categorical](screenshots/04-categorical-features.jpg)

- **November: 25.4%** conversion (holiday peak)
- New visitors convert better (24.9%) than returning (13.9%)
- Weekend: 17.4% vs Weekday: 14.9%

### Traffic Performance

![Traffic](screenshots/05-traffic-types.jpg)

**Best sources:**
- Type 8: 27.7% conversion (partnerships)
- Type 2: 21.6% (organic search)
- Type 5: 21.5% (email)

**Volume leader:** Type 2 with 4,000+ sessions

---

## Actions

| Probability | Intent | Action |
|------------|--------|--------|
| 70%+ | 🔥 High | Show 10-15% discount |
| 40-70% | ⚡ Medium | Cart reminder email |
| 20-40% | 💭 Low | Retarget ads 7 days |
| <20% | ❄️ Very Low | No action |

**Recommendations:**
- Double ad spend Nov/Dec (2x conversion)
- Focus on Traffic Type 8 partnerships (27.7% conversion)
- Target visitors with PageValues > 10 (4.5x more likely to buy)

---

## Stack

Python 3.13 • XGBoost • Scikit-learn • Flask • Docker • Pandas • NumPy

---

## Files
```
ecommerce_predicition/
├── data/
│ └── online_shoppers_intention.csv
├── screenshots/
│ └── [6 images]
├── online_shoppers.ipynb # Full analysis
├── train.py # Training
├── predict.py # Flask API
├── model.json # Trained model
├── dv.pkl # Encoder
├── Pipfile
├── Dockerfile
└── README.md
```



