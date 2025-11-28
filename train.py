import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def load_data(filepath='online_shoppers_intention.csv'):
    df = pd.read_csv(filepath)
    return df

def engineer_features(df):
    # Total pages
    df['TotalPages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
    
    # Total duration
    df['TotalDuration'] = (df['Administrative_Duration'] + 
                           df['Informational_Duration'] + 
                           df['ProductRelated_Duration'])
    
    # Average time per page
    df['AvgTimePerPage'] = df['TotalDuration'] / (df['TotalPages'] + 1)
    
    # Product page ratio
    df['ProductPageRatio'] = df['ProductRelated'] / (df['TotalPages'] + 1)
    
    # High value visitor
    df['IsHighValueVisitor'] = (df['PageValues'] > df['PageValues'].median()).astype(int)
    
    # Holiday season
    df['IsHolidaySeason'] = df['Month'].isin(['Nov', 'Dec']).astype(int)
    
    # Engagement score
    df['EngagementScore'] = (
        (df['TotalPages'] / df['TotalPages'].max()) * 0.3 +
        (df['TotalDuration'] / df['TotalDuration'].max()) * 0.3 +
        ((1 - df['BounceRates']) * 0.2) +
        ((1 - df['ExitRates']) * 0.2)
    )
    
    return df

def prepare_data(df):
    df['Revenue'] = df['Revenue'].astype(int)
    
    X = df.drop(['Revenue'], axis=1)
    y = df['Revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def encode_features(X_train, X_test):
    train_dicts = X_train.fillna(0).to_dict(orient='records')
    test_dicts = X_test.fillna(0).to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train_enc = dv.fit_transform(train_dicts)
    X_test_enc = dv.transform(test_dicts)
    
    return X_train_enc, X_test_enc, dv

def train_model(X_train, y_train):
    """Train XGBoost model"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.3,
        'seed': 42
    }
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

def main():
   
    # Load data
    print("\nLoading data")
    df = load_data()
    print(f"Loaded {len(df):,} sessions")
    
    # Engineer features
    print("\nEngineering features")
    df = engineer_features(df)
    
    # Prepare data
    print("\nSplitting data")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Encode features
    print("\nEncoding features")
    X_train_enc, X_test_enc, dv = encode_features(X_train, X_test)
    print(f"   Encoded features: {X_train_enc.shape[1]}")
    
    # Train model
    print("\nTraining XGBoost model")
    model = train_model(X_train_enc, y_train)
    print(f"   Model trained")
    
    # Evaluate
    print("\nEvaluating on test set")
    dtest = xgb.DMatrix(X_test_enc)
    y_pred = model.predict(dtest)
    auc = roc_auc_score(y_test, y_pred)
    print(f"   Test AUC: {auc:.4f}")
    
    # Save
    print("\nSaving model")
    model.save_model('model.json')
    with open('dv.pkl', 'wb') as f:
        pickle.dump(dv, f)

    print(f"\n Final Model AUC: {auc:.4f}")

if __name__ == '__main__':
    main()