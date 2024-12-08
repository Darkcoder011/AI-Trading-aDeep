from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flaml import AutoML
from optuna.integration import OptunaSearchCV
from skopt import BayesSearchCV
import ta
import io
from datetime import datetime, timedelta
import h2o
from h2o.automl import H2OAutoML

# Initialize H2O
h2o.init()

router = APIRouter()

class MarketModel:
    def __init__(self, name, model, emoji, specialization):
        self.name = name
        self.model = model
        self.emoji = emoji
        self.specialization = specialization

def prepare_market_features(df):
    # Add technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['bb_high'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
    df['bb_low'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
    df['volume_sma'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    # Calculate price changes
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Create target variable (1 for uptrend, 0 for downtrend/sideways)
    df['target'] = (df['close'].shift(-10) > df['close']).astype(int)
    
    return df.dropna()

def calculate_pressure(df):
    volume = df['volume'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # Calculate buying pressure
    buying_pressure = np.sum((close - low) * volume) / np.sum(volume)
    
    # Calculate selling pressure
    selling_pressure = np.sum((high - close) * volume) / np.sum(volume)
    
    # Normalize pressures
    total_pressure = buying_pressure + selling_pressure
    buying_pressure = buying_pressure / total_pressure
    selling_pressure = selling_pressure / total_pressure
    
    return buying_pressure, selling_pressure

def prepare_data(df):
    # Prepare features
    df = prepare_market_features(df)
    
    feature_columns = ['rsi', 'macd', 'bb_high', 'bb_low', 'volume_sma', 'obv', 
                      'price_change', 'volume_change']
    
    X = df[feature_columns]
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler, X.iloc[-1:]

def get_market_models():
    # Create base classifier for Optuna
    base_classifier = RandomForestClassifier(random_state=42)
    optuna_search = OptunaSearchCV(
        base_classifier,
        {
            'n_estimators': (10, 100),
            'max_depth': (3, 10),
            'min_samples_split': (2, 10)
        },
        n_trials=10,
        random_state=42
    )

    # Create base classifier for Bayesian Optimization
    bayes_search = BayesSearchCV(
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': (10, 100),
            'max_depth': (3, 10),
            'min_samples_split': (2, 10)
        },
        n_iter=10,
        random_state=42
    )

    models = [
        MarketModel("TrendMaster XGB", xgb.XGBClassifier(random_state=42), "🎯", "Overall Trend"),
        MarketModel("VolumeWave LGB", lgb.LGBMClassifier(random_state=42), "📊", "Volume Analysis"),
        MarketModel("MomentumCat", CatBoostClassifier(random_state=42, verbose=False), "🌊", "Momentum"),
        MarketModel("PriceForest", RandomForestClassifier(random_state=42), "🌳", "Price Patterns"),
        MarketModel("WaveRider GBM", GradientBoostingClassifier(random_state=42), "🏄", "Wave Analysis"),
        MarketModel("NeuralMarket", MLPClassifier(random_state=42), "🧠", "Pattern Recognition"),
        MarketModel("SupportVector Pro", SVC(probability=True, random_state=42), "📐", "Support/Resistance"),
        MarketModel("TrendForce XGB", xgb.XGBClassifier(random_state=42), "💪", "Trend Strength"),
        MarketModel("VolumePulse LGB", lgb.LGBMClassifier(random_state=42), "💫", "Volume Patterns"),
        MarketModel("MarketMind Cat", CatBoostClassifier(random_state=42, verbose=False), "🔮", "Market Psychology"),
        MarketModel("H2O DeepLearning Pro", MLPClassifier(random_state=42), "🤖", "Deep Pattern Analysis"),
        MarketModel("H2O GBM Expert", GradientBoostingClassifier(random_state=42), "🎓", "Advanced Gradient Boosting"),
        MarketModel("H2O AutoML Master", VotingClassifier([
            ('xgb', xgb.XGBClassifier(random_state=42)),
            ('lgb', lgb.LGBMClassifier(random_state=42)),
            ('cat', CatBoostClassifier(random_state=42, verbose=False))
        ]), "⚡", "Automated Model Selection"),
        # New AutoML models (Windows compatible)
        MarketModel("FLAML Expert", AutoML(), "🌟", "Fast & Light AutoML"),
        MarketModel("Optuna Master", optuna_search, "🔮", "Hyperparameter Optimization"),
        MarketModel("Bayesian Optimizer", bayes_search, "🎯", "Smart Parameter Search")
    ]
    return models

def predict_trend(probability):
    if probability > 0.7:
        return "strong_uptrend"
    elif probability > 0.55:
        return "weak_uptrend"
    elif probability < 0.3:
        return "strong_downtrend"
    elif probability < 0.45:
        return "weak_downtrend"
    else:
        return "sideways"

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, X_last):
    try:
        if isinstance(model.model, AutoML):
            # FLAML training
            model.model.fit(X_train, y_train, task='classification', time_budget=30)
            y_pred = model.model.predict(X_test)
            y_pred_proba = model.model.predict_proba(X_test)[:, 1]
            next_prediction_prob = float(model.model.predict_proba(X_last)[:, 1])
        elif isinstance(model.model, (OptunaSearchCV, BayesSearchCV)):
            # Optuna and Bayesian optimization
            model.model.fit(X_train, y_train)
            y_pred = model.model.predict(X_test)
            y_pred_proba = model.model.predict_proba(X_test)[:, 1]
            next_prediction_prob = float(model.model.predict_proba(X_last)[:, 1])
        else:
            # Regular sklearn-style models
            model.model.fit(X_train, y_train)
            y_pred = model.model.predict(X_test)
            y_pred_proba = model.model.predict_proba(X_test)[:, 1]
            next_prediction_prob = float(model.model.predict_proba(X_last)[:, 1])
        
        next_prediction = predict_trend(next_prediction_prob)
        
        return {
            'name': model.name,
            'emoji': model.emoji,
            'specialization': model.specialization,
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'next_prediction': next_prediction,
            'confidence': next_prediction_prob
        }
    except Exception as e:
        print(f"Error training {model.name}: {str(e)}")
        return {
            'name': model.name,
            'emoji': model.emoji,
            'specialization': model.specialization,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'next_prediction': 'sideways',
            'confidence': 0.5
        }

def calculate_consensus_prediction(model_predictions):
    prediction_weights = {
        'strong_uptrend': 2,
        'weak_uptrend': 1,
        'sideways': 0,
        'weak_downtrend': -1,
        'strong_downtrend': -2
    }
    
    weighted_sum = 0
    total_confidence = 0
    
    for pred in model_predictions:
        weight = prediction_weights[pred['next_prediction']]
        confidence = pred['confidence']
        weighted_sum += weight * confidence
        total_confidence += confidence
    
    average_score = weighted_sum / total_confidence
    
    if average_score > 1:
        return "strong_uptrend"
    elif average_score > 0.2:
        return "weak_uptrend"
    elif average_score < -1:
        return "strong_downtrend"
    elif average_score < -0.2:
        return "weak_downtrend"
    else:
        return "sideways"

def calculate_combined_predictions(model_predictions):
    total_models = len(model_predictions)
    trend_counts = {
        'strong_uptrend': 0,
        'weak_uptrend': 0,
        'sideways': 0,
        'weak_downtrend': 0,
        'strong_downtrend': 0
    }
    
    # Count predictions
    for pred in model_predictions:
        if pred['next_prediction'] in trend_counts:
            trend_counts[pred['next_prediction']] += 1
    
    # Calculate percentages and confidences
    trend_analysis = []
    for trend, count in trend_counts.items():
        percentage = (count / total_models) * 100
        # Get average confidence of models predicting this trend
        confidence = np.mean([
            pred['confidence']
            for pred in model_predictions
            if pred['next_prediction'] == trend
        ]) if count > 0 else 0
        
        trend_analysis.append({
            'trend': trend,
            'count': count,
            'percentage': percentage,
            'confidence': confidence,
            'models': [
                {'name': pred['name'], 'emoji': pred['emoji'], 'confidence': pred['confidence']}
                for pred in model_predictions
                if pred['next_prediction'] == trend
            ]
        })
    
    # Sort by count (highest first)
    trend_analysis.sort(key=lambda x: (x['count'], x['confidence']), reverse=True)
    
    return {
        'total_models': total_models,
        'trends': trend_analysis,
        'dominant_trend': trend_analysis[0]['trend'] if trend_analysis else 'sideways',
        'highest_confidence': max((item['confidence'] for item in trend_analysis), default=0)
    }

@router.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    try:
        # Read the CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Calculate buying/selling pressure
        buying_pressure, selling_pressure = calculate_pressure(df)
        
        # Prepare the data
        X_train, X_test, y_train, y_test, feature_names, scaler, X_last = prepare_data(df)
        
        # Get model predictions
        models = get_market_models()
        model_predictions = []
        
        for model in models:
            prediction = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, scaler.transform(X_last))
            model_predictions.append(prediction)
        
        # Calculate consensus prediction
        consensus = calculate_consensus_prediction(model_predictions)
        
        # Calculate combined predictions analysis
        combined_predictions = calculate_combined_predictions(model_predictions)
        
        # Calculate feature importance using XGBoost
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train)
        feature_importance = [
            {'name': name, 'importance': float(importance)}
            for name, importance in zip(feature_names, xgb_model.feature_importances_)
        ]
        
        # Prepare next timestamp
        next_time = datetime.now() + timedelta(minutes=10)
        
        results = {
            'models': model_predictions,
            'feature_importance': sorted(feature_importance, key=lambda x: x['importance'], reverse=True),
            'pressure_analysis': {
                'buying_pressure': float(buying_pressure),
                'selling_pressure': float(selling_pressure),
                'net_pressure': float(buying_pressure - selling_pressure)
            },
            'consensus_prediction': {
                'trend': consensus,
                'next_timestamp': next_time.strftime('%Y-%m-%d %H:%M:%S'),
                'confidence_level': sum(m['confidence'] for m in model_predictions) / len(model_predictions)
            },
            'combined_predictions': combined_predictions
        }
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
