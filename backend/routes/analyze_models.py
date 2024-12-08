from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, AdaBoostClassifier, BaggingClassifier,
    HistGradientBoostingClassifier, IsolationForest
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    PassiveAggressiveClassifier, Perceptron
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from prophet import Prophet
import arch
import json
import sys
import os
import ta
from io import StringIO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.indicators import calculate_rsi

router = APIRouter()

def create_features(df):
    """Create advanced technical features for better 15-minute predictions"""
    # Convert timestamp to datetime if needed
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Sort by datetime to ensure correct order
    df = df.sort_values('datetime')
    
    # Basic Features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['range'] = (df['high'] - df['low']) / df['close']
    df['price_acceleration'] = df['returns'].diff()
    
    # Handle NaN values using forward fill first, then backward fill
    df = df.ffill().bfill()
    
    # Only calculate technical indicators if we have enough data
    if len(df) >= 20:  # Minimum required for most indicators
        try:
            # Volume Features
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_std'] = df['volume'].rolling(window=20).std()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            # Momentum Indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
            df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], window=14)
            df['ultimate_oscillator'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
            
            # Trend Indicators
            df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)
            df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
            df['dpo'] = ta.trend.dpo(df['close'])
            df['ichimoku_a'] = ta.trend.ichimoku_a(df['high'], df['low'])
            df['ichimoku_b'] = ta.trend.ichimoku_b(df['high'], df['low'])
            
            # Volatility Indicators
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            df['keltner_upper'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'])
            df['keltner_lower'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'])
            
            # Pattern Recognition
            df['engulfing'] = ta.candlestick.cdl_engulfing(df['open'], df['high'], df['low'], df['close'])
            df['morning_star'] = ta.candlestick.cdl_morning_star(df['open'], df['high'], df['low'], df['close'])
            df['evening_star'] = ta.candlestick.cdl_evening_star(df['open'], df['high'], df['low'], df['close'])
            df['doji'] = ta.candlestick.cdl_doji(df['open'], df['high'], df['low'], df['close'])
            
            # Custom Features
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['hl_intensity'] = abs(df['high'] - df['low']) / (abs(df['close'] - df['open']) + 0.001)
            df['volume_intensity'] = df['volume'] / df['volume'].rolling(window=10).mean()
            df['trend_intensity'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            
        except Exception as e:
            print(f"Warning: Error calculating some technical indicators: {str(e)}")
    
    # Target Variable (15-min ahead prediction)
    df['target'] = (df['close'].shift(-3) > df['close']).astype(int)  # 3 periods of 5m = 15 minutes
    
    # Final NaN handling
    df = df.ffill().bfill()
    
    return df

@router.post("/analyze_models")
async def analyze_models(file: UploadFile = File(...)):
    try:
        print("Starting model analysis...")
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Ensure minimum data requirements
        if len(df) < 20:  # Reduced minimum requirement
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data. Need at least 20 candles, current: {len(df)}"
            )

        # Create advanced features
        df = create_features(df)
        
        # Prepare features for training
        feature_columns = [col for col in df.columns if col not in ['target', 'timestamp', 'datetime', 'timeframe']]
        X = df[feature_columns].iloc[:-3]  # Remove last 3 points (15 minutes) as they don't have targets
        y = df['target'].iloc[:-3]
        
        # Adjust TimeSeriesSplit based on data size
        n_splits = min(5, len(X) // 10)  # Ensure we don't split too many times
        test_size = min(20, len(X) // 5)  # Adjust test size based on data size
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        split = list(tscv.split(X))
        train_idx, test_idx = split[-1]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features using RobustScaler for better handling of outliers
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to DataFrame to preserve feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Define all models with optimized parameters
        models = {
            # Original Models (Kept as is)
            'RandomForest': RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=300, max_depth=5),
            'CatBoost': CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, verbose=0),
            'KNeighbors': KNeighborsClassifier(n_neighbors=5),
            'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
            'GaussianNB': GaussianNB(),
            'SVC': SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42),
            'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=200, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=300, random_state=42),
            'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
            'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'RidgeClassifier': RidgeClassifier(),

            # New Stock Market Specific Models
            'LightGBM': LGBMClassifier(
                n_estimators=200, 
                learning_rate=0.01,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'ProphetModel': None,  # Handled separately with FB Prophet
            'LSTM': None,  # Handled separately with deep learning
            'GARCH': None,  # Handled separately for volatility
            'ElasticNet': SGDClassifier(
                loss='modified_huber',
                penalty='elasticnet',
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=1000,
                random_state=42
            ),
            'VotingEnsemble': VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                    ('xgb', XGBClassifier(n_estimators=200, random_state=42)),
                    ('lgb', LGBMClassifier(n_estimators=200, random_state=42))
                ],
                voting='soft'
            ),
            'StackingEnsemble': StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                    ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
                    ('cat', CatBoostClassifier(iterations=100, random_state=42))
                ],
                final_estimator=LogisticRegression()
            ),
            'IsolationForest': IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            ),
            'QuadraticDA': QuadraticDiscriminantAnalysis(
                reg_param=0.1,
                store_covariance=True
            ),
            'LinearDA': LinearDiscriminantAnalysis(
                solver='svd',
                shrinkage=None
            ),
        }
        
        results = {}
        ensemble_predictions = []
        
        # Train and evaluate each model
        for name, model in models.items():
            try:
                if model is not None:  # Skip special models
                    print(f"Training {name}...")
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test_scaled)[-1]
                    else:
                        y_prob = np.array([0.5, 0.5])  # Default probabilities
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    
                    results[name] = {
                        'accuracy': float(accuracy * 100),
                        'f1': float(f1 * 100),
                        'precision': float(precision * 100),
                        'recall': float(recall * 100),
                        'prediction': bool(y_prob[1] > 0.5),
                        'probability': {
                            'down': float(y_prob[0]),
                            'up': float(y_prob[1])
                        }
                    }
                    ensemble_predictions.append(y_prob[1] > 0.5)
                    print(f"{name} completed. Accuracy: {accuracy:.2f}")
                    
            except Exception as e:
                print(f"Error in {name}: {str(e)}")
                continue
        
        # Special handling for Prophet
        try:
            print("Training Prophet...")
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['datetime']),
                'y': df['close']
            })
            
            # Train Prophet model
            prophet = Prophet(daily_seasonality=True)
            prophet.fit(prophet_df)
            
            # Make future dataframe for prediction
            future = prophet.make_future_dataframe(periods=1, freq='5min')
            forecast = prophet.predict(future)
            
            # Get the prediction
            last_prediction = forecast['yhat'].iloc[-1]
            last_actual = prophet_df['y'].iloc[-1]
            prediction = last_prediction > last_actual
            
            # Calculate accuracy on test set
            historical_predictions = (forecast['yhat'].iloc[:-1].values > prophet_df['y'].values).astype(int)
            accuracy = accuracy_score(y, historical_predictions[-len(y):])
            f1 = f1_score(y, historical_predictions[-len(y):], average='weighted')
            
            results['Prophet'] = {
                'accuracy': float(accuracy),
                'f1': float(f1),
                'probability': {
                    'up': float(last_prediction > last_actual),
                    'down': float(last_prediction <= last_actual)
                }
            }
            print(f"Prophet completed. Accuracy: {accuracy:.2f}")
        except Exception as e:
            print(f"Error in Prophet: {str(e)}")

        # Ensure all model results are JSON serializable
        for model_name in results:
            if 'prediction' in results[model_name]:
                results[model_name]['prediction'] = bool(results[model_name]['prediction'])
            
            # Convert numpy types to Python native types
            if 'accuracy' in results[model_name]:
                results[model_name]['accuracy'] = float(results[model_name]['accuracy'])
            if 'f1' in results[model_name]:
                results[model_name]['f1'] = float(results[model_name]['f1'])
            
            # Ensure probabilities are native Python floats
            if 'probability' in results[model_name]:
                results[model_name]['probability'] = {
                    'up': float(results[model_name]['probability']['up']),
                    'down': float(results[model_name]['probability']['down'])
                }

        # Special handling for LSTM
        try:
            print("Training LSTM...")
            # Reshape data for LSTM [samples, timesteps, features]
            X_train_lstm = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_lstm = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
            
            model = Sequential([
                Input(shape=(1, X_train.shape[1])),
                LSTM(50, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
            
            y_pred = (model.predict(X_test_lstm) > 0.5).astype(int).flatten()
            proba = model.predict(X_test_lstm).flatten()
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results['LSTM'] = {
                'accuracy': accuracy,
                'f1': f1,
                'probability': {
                    'up': float(np.mean(proba)),
                    'down': float(1 - np.mean(proba))
                }
            }
            print(f"LSTM completed. Accuracy: {accuracy:.2f}")
        except Exception as e:
            print(f"Error in LSTM: {str(e)}")

        # Special handling for GARCH
        try:
            print("Training GARCH...")
            # Scale the returns for GARCH
            returns = df['returns'].values * 10000  # Scale up by 10000 as recommended
            
            # Fit GARCH model
            model = arch.arch_model(returns, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            
            # Use volatility forecast as a trading signal
            forecast = model_fit.forecast(horizon=1)
            volatility = forecast.variance.values[-1, 0]
            
            # Convert volatility to binary prediction (high volatility = 1, low volatility = 0)
            threshold = np.median(returns**2)
            y_pred = (volatility > threshold).astype(int)
            
            # Calculate metrics
            accuracy = float(y_pred == y_test[-1])  # Compare with last actual value
            results['GARCH'] = {
                'accuracy': accuracy,
                'f1': accuracy,  # For single prediction, accuracy equals F1
                'probability': {
                    'up': float(volatility > threshold),
                    'down': float(volatility <= threshold)
                }
            }
            print(f"GARCH completed. Accuracy: {accuracy:.2f}")
        except Exception as e:
            print(f"Error in GARCH: {str(e)}")

        # Train IsolationForest
        try:
            print("Training IsolationForest...")
            iso_forest = IsolationForest(random_state=42, contamination=0.1)
            iso_forest.fit(X_train)
            # Convert predictions from 1/-1 to 1/0
            y_pred = (iso_forest.predict(X_test) == 1).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results['IsolationForest'] = {
                'accuracy': accuracy,
                'f1': f1,
                'probability': {
                    'up': float(np.mean(y_pred)),
                    'down': float(1 - np.mean(y_pred))
                }
            }
            print(f"IsolationForest completed. Accuracy: {accuracy:.2f}")
        except Exception as e:
            print(f"Error in IsolationForest: {str(e)}")

        # Train QuadraticDA with error handling for collinearity
        try:
            print("Training QuadraticDA...")
            # Add small regularization to handle collinearity
            qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
            qda.fit(X_train, y_train)
            y_pred = qda.predict(X_test)
            proba = qda.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results['QuadraticDA'] = {
                'accuracy': accuracy,
                'f1': f1,
                'probability': {
                    'up': float(np.mean(proba[:, 1])),
                    'down': float(np.mean(proba[:, 0]))
                }
            }
            print(f"QuadraticDA completed. Accuracy: {accuracy:.2f}")
        except Exception as e:
            print(f"Error in QuadraticDA: {str(e)}")

        # Train LinearDA
        try:
            print("Training LinearDA...")
            lda = LinearDiscriminantAnalysis(solver='svd')
            lda.fit(X_train, y_train)
            y_pred = lda.predict(X_test)
            proba = lda.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results['LinearDA'] = {
                'accuracy': accuracy,
                'f1': f1,
                'probability': {
                    'up': float(np.mean(proba[:, 1])),
                    'down': float(np.mean(proba[:, 0]))
                }
            }
            print(f"LinearDA completed. Accuracy: {accuracy:.2f}")
        except Exception as e:
            print(f"Error in LinearDA: {str(e)}")

        # Ensure all results have required fields
        for model_name in list(results.keys()):
            if 'accuracy' not in results[model_name] or 'f1' not in results[model_name] or 'probability' not in results[model_name]:
                print(f"Removing incomplete results for {model_name}")
                del results[model_name]

        # Calculate weighted ensemble prediction with optimized weights
        model_weights = {
            # Gradient Boosting Models (Higher weights due to better performance with financial data)
            'XGBoost': 0.12,
            'LightGBM': 0.12,
            'CatBoost': 0.12,
            'GradientBoosting': 0.10,
            
            # Ensemble Models
            'RandomForest': 0.08,
            'ExtraTrees': 0.08,
            'VotingEnsemble': 0.08,
            'StackingEnsemble': 0.08,
            
            # Specialized Models
            'LSTM': 0.06,
            'Prophet': 0.06,
            'GARCH': 0.06,
            
            # Base Models (Lower weights due to simplicity)
            'AdaBoost': 0.02,
            'Bagging': 0.02,
            'KNeighbors': 0.02,
            'MLP': 0.02,
            'SVC': 0.02,
            'LogisticRegression': 0.02,
            'DecisionTree': 0.02,
            'RidgeClassifier': 0.02,
            'ElasticNet': 0.02,
            'IsolationForest': 0.02,
            'QuadraticDA': 0.02,
            'LinearDA': 0.02,
            'HistGradientBoosting': 0.02,
            'GaussianNB': 0.02
        }

        # Normalize weights to ensure they sum to 1
        total_weight = sum(model_weights.values())
        model_weights = {k: v/total_weight for k, v in model_weights.items()}

        # Calculate weighted prediction with performance adjustment
        weighted_prediction = 0
        total_applied_weight = 0

        for name, result in results.items():
            if name in model_weights:
                # Adjust weight based on model performance
                performance_score = (result['accuracy'] + result['f1']) / 200  # Normalize to 0-1
                adjusted_weight = model_weights[name] * performance_score
                
                # Add to weighted prediction
                if 'probability' in result:
                    weighted_prediction += result['probability']['up'] * adjusted_weight
                    total_applied_weight += adjusted_weight

        # Normalize the final prediction
        if total_applied_weight > 0:
            weighted_prediction /= total_applied_weight

        # More aggressive decision making
        decision = 'HOLD'
        confidence_threshold_strong = 0.65  # 65% for strong signals
        confidence_threshold_moderate = 0.55  # 55% for moderate signals

        up_percentage = weighted_prediction * 100
        down_percentage = (1 - weighted_prediction) * 100
        confidence_level = abs(up_percentage - down_percentage)

        # Dynamic thresholds based on market volatility
        if 'volatility' in df.columns:
            recent_volatility = df['volatility'].iloc[-1]
            volatility_adjustment = min(0.05, recent_volatility)  # Cap at 5%
            confidence_threshold_strong -= volatility_adjustment * 100
            confidence_threshold_moderate -= volatility_adjustment * 100

        # Add market context to the decision
        class MarketContextAnalyzer:
            def __init__(self, df):
                self.df = df
                self.current_close = df['close'].iloc[-1]
                self.current_high = df['high'].iloc[-1]
                self.current_low = df['low'].iloc[-1]
                self.current_volume = df['volume'].iloc[-1] if 'volume' in df.columns else None
                
            def analyze_trend(self):
                try:
                    # Multiple timeframe analysis
                    sma10 = self.df['close'].rolling(window=10).mean()
                    sma20 = self.df['close'].rolling(window=20).mean()
                    sma50 = self.df['close'].rolling(window=50).mean()
                    
                    # Current values
                    curr_sma10 = sma10.iloc[-1]
                    curr_sma20 = sma20.iloc[-1]
                    curr_sma50 = sma50.iloc[-1]
                    
                    # Price momentum
                    momentum = (self.df['close'].diff(5).iloc[-1] / self.df['close'].iloc[-5]) * 100
                    
                    if self.current_close > curr_sma10 > curr_sma20 > curr_sma50 and momentum > 0.5:
                        return "strong_uptrend"
                    elif self.current_close < curr_sma10 < curr_sma20 < curr_sma50 and momentum < -0.5:
                        return "strong_downtrend"
                    elif self.current_close > curr_sma20 and momentum > 0:
                        return "uptrend"
                    elif self.current_close < curr_sma20 and momentum < 0:
                        return "downtrend"
                    else:
                        return "neutral"
                except Exception as e:
                    print(f"Trend analysis error: {str(e)}")
                    return "neutral"
                    
            def analyze_volume(self):
                try:
                    if self.current_volume is None:
                        return False
                        
                    # Volume analysis
                    vol_sma20 = self.df['volume'].rolling(window=20).mean()
                    vol_sma5 = self.df['volume'].rolling(window=5).mean()
                    
                    # Volume conditions
                    vol_ratio = self.current_volume / vol_sma20.iloc[-1]
                    vol_increasing = vol_sma5.iloc[-1] > vol_sma20.iloc[-1]
                    vol_spike = vol_ratio > 1.5
                    
                    return vol_increasing and vol_spike
                except Exception as e:
                    print(f"Volume analysis error: {str(e)}")
                    return False
                    
            def analyze_rsi(self):
                try:
                    # RSI Calculation
                    delta = self.df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # RSI with price action confirmation
                    price_higher = self.df['close'].iloc[-1] > self.df['close'].iloc[-2]
                    
                    if current_rsi < 30 and price_higher:
                        return "strong_buy"
                    elif current_rsi > 70 and not price_higher:
                        return "strong_sell"
                    elif current_rsi < 40:
                        return "buy"
                    elif current_rsi > 60:
                        return "sell"
                    else:
                        return "neutral"
                except Exception as e:
                    print(f"RSI analysis error: {str(e)}")
                    return "neutral"
                    
            def analyze_volatility(self):
                try:
                    # True Range calculation
                    high_low = self.df['high'] - self.df['low']
                    high_close = abs(self.df['high'] - self.df['close'].shift())
                    low_close = abs(self.df['low'] - self.df['close'].shift())
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    
                    # ATR (14 periods)
                    atr = tr.rolling(window=14).mean()
                    
                    # Volatility as percentage of price
                    volatility = (atr.iloc[-1] / self.current_close) * 100
                    
                    # Normalize between 0 and 100
                    max_vol = atr.rolling(window=100).max().iloc[-1] / self.current_close * 100
                    min_vol = atr.rolling(window=100).min().iloc[-1] / self.current_close * 100
                    
                    if max_vol == min_vol:
                        return 0.0
                        
                    normalized_vol = ((volatility - min_vol) / (max_vol - min_vol)) * 100
                    return round(normalized_vol, 2)
                except Exception as e:
                    print(f"Volatility analysis error: {str(e)}")
                    return 0.0

        # Calculate market context using the analyzer
        try:
            context_analyzer = MarketContextAnalyzer(df)
            trend_conf = context_analyzer.analyze_trend()
            vol_conf = context_analyzer.analyze_volume()
            rsi_sig = context_analyzer.analyze_rsi()
            vol_adj = context_analyzer.analyze_volatility()
            
            market_context = {
                'trend_confirmation': trend_conf,
                'volume_confirmation': vol_conf,
                'rsi_signal': rsi_sig,
                'volatility_adjustment': vol_adj
            }
        except Exception as e:
            print(f"Market context analysis error: {str(e)}")
            market_context = {
                'trend_confirmation': 'neutral',
                'volume_confirmation': False,
                'rsi_signal': 'neutral',
                'volatility_adjustment': 0.0
            }

        # Trend confirmation
        recent_trend = None
        if 'ema_5' in df.columns and 'ema_20' in df.columns:
            ema_5_trend = df['ema_5'].iloc[-1] > df['ema_5'].iloc[-2]
            ema_20_trend = df['ema_20'].iloc[-1] > df['ema_20'].iloc[-2]
            if ema_5_trend and ema_20_trend:
                recent_trend = 'up'
            elif not ema_5_trend and not ema_20_trend:
                recent_trend = 'down'

        # Volume confirmation
        volume_confirmed = False
        if 'volume_ratio' in df.columns:
            volume_confirmed = df['volume_ratio'].iloc[-1] > 1.0

        # RSI extremes check
        rsi_signal = None
        if 'rsi' in df.columns:
            rsi_value = df['rsi'].iloc[-1]
            if rsi_value < 30:
                rsi_signal = 'up'
            elif rsi_value > 70:
                rsi_signal = 'down'

        # Decision making with multiple confirmations
        if confidence_level >= confidence_threshold_strong:
            if up_percentage > down_percentage:
                decision = 'STRONG_BUY' if (recent_trend == 'up' or rsi_signal == 'up') and volume_confirmed else 'BUY'
            else:
                decision = 'STRONG_SELL' if (recent_trend == 'down' or rsi_signal == 'down') and volume_confirmed else 'SELL'
        elif confidence_level >= confidence_threshold_moderate:
            if up_percentage > down_percentage:
                decision = 'BUY' if recent_trend == 'up' or rsi_signal == 'up' else 'HOLD'
            else:
                decision = 'SELL' if recent_trend == 'down' or rsi_signal == 'down' else 'HOLD'

        ensemble_decision = {
            'decision': str(decision),  # Ensure decision is a string
            'up_probability': float(up_percentage),
            'down_probability': float(down_percentage),
            'confidence_level': float(confidence_level),
            'model_weights': {k: float(v) for k, v in model_weights.items()},  # Convert weights to float
            'market_context': {
                'trend_confirmation': str(recent_trend) if recent_trend else None,
                'volume_confirmation': bool(volume_confirmed),
                'rsi_signal': str(rsi_signal) if rsi_signal else None,
                'volatility_adjustment': float(volatility_adjustment) if 'volatility_adjustment' in locals() else None
            },
            'prediction_timeframe': '15 minutes'
        }

        # Final response with all JSON serializable values
        response = {
            'ensemble_decision': ensemble_decision,
            'model_predictions': results
        }

        # Print individual model results
        print("\nIndividual Model Results:")
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            if 'accuracy' in result:
                print(f"Accuracy: {result['accuracy']:.2f}")
            if 'f1' in result:
                print(f"F1 Score: {result['f1']:.2f}")
            if 'probability' in result:
                print(f"Up Probability: {result['probability']['up']:.2f}")
                print(f"Down Probability: {result['probability']['down']:.2f}")

        # Ensure all model results are JSON serializable and complete
        processed_results = {}
        for model_name, result in results.items():
            processed_result = {
                'accuracy': float(result.get('accuracy', 0)),
                'f1': float(result.get('f1', 0)),
                'probability': {
                    'up': float(result.get('probability', {}).get('up', 0)),
                    'down': float(result.get('probability', {}).get('down', 0))
                },
                'prediction': 'UP' if result.get('probability', {}).get('up', 0) > 0.5 else 'DOWN'
            }
            processed_results[model_name] = processed_result

        # Calculate feature importance for applicable models
        feature_importance = {}
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_columns, model.feature_importances_))
                feature_importance[model_name] = {k: float(v) for k, v in importance.items()}

        # Print ensemble decision details
        print("\nEnsemble Decision Details:")
        print(f"Final Decision: {decision}")
        print(f"Up Probability: {up_percentage:.2f}%")
        print(f"Down Probability: {down_percentage:.2f}%")
        print(f"Confidence Level: {confidence_level:.2f}%")

        # Create detailed response
        detailed_response = {
            'ensemble_decision': {
                'decision': str(decision),
                'up_probability': float(up_percentage),
                'down_probability': float(down_percentage),
                'confidence_level': float(confidence_level),
                'prediction_timeframe': '15 minutes'
            },
            'model_predictions': processed_results,
            'model_weights': {k: float(v) for k, v in model_weights.items()},  # Convert weights to float
            'market_analysis': {
                'trend_confirmation': str(recent_trend) if recent_trend else None,
                'volume_confirmation': bool(volume_confirmed),
                'rsi_signal': str(rsi_signal) if rsi_signal else None,
                'volatility_adjustment': float(volatility_adjustment) if 'volatility_adjustment' in locals() else None
            },
            'technical_indicators': {
                'current_rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else None,
                'current_macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else None,
                'current_volume_ratio': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else None,
                'current_volatility': float(df['volatility'].iloc[-1]) if 'volatility' in df.columns else None
            },
            'metadata': {
                'total_candles': len(df),
                'training_size': len(X_train),
                'test_size': len(X_test),
                'features_used': list(feature_columns),
                'feature_importance': feature_importance
            }
        }

        print("\nAnalysis completed successfully!")
        return detailed_response

    except Exception as e:
        print(f"Error in analyze_models: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def get_feature_importance(models, feature_names):
    """Calculate feature importance across all models that support it"""
    feature_importance = {}
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            for fname, imp in zip(feature_names, importance):
                if fname not in feature_importance:
                    feature_importance[fname] = []
                feature_importance[fname].append(imp)
    
    # Average importance across models
    avg_importance = {
        fname: float(np.mean(scores))
        for fname, scores in feature_importance.items()
    }
    
    # Sort by importance
    sorted_importance = dict(sorted(
        avg_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20])  # Top 20 features
    
    return sorted_importance
