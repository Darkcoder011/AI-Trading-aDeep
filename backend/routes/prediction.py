from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier
import ta
from io import StringIO
import sys
import os
import logging
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.indicators import calculate_rsi

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

def calculate_advanced_indicators(df):
    try:
        # Volume-based indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ema'] = df['volume'].ewm(span=20, adjust=False).mean()
        df['force_index'] = ta.volume.force_index(df['close'], df['volume'], window=13)
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        
        # Trend indicators
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        df['dpo'] = ta.trend.dpo(df['close'], window=20)
        
        # Moving Averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # DMI calculation using alternative method
        # Calculate True Range first
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate +DM and -DM
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        df['pos_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'],
            0
        )
        df['neg_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'],
            0
        )
        
        # Calculate smoothed values
        window = 14
        df['tr_smoothed'] = df['tr'].rolling(window=window).mean()
        df['pos_dm_smoothed'] = df['pos_dm'].rolling(window=window).mean()
        df['neg_dm_smoothed'] = df['neg_dm'].rolling(window=window).mean()
        
        # Calculate +DI and -DI
        df['dmi_plus'] = 100 * (df['pos_dm_smoothed'] / df['tr_smoothed'])
        df['dmi_minus'] = 100 * (df['neg_dm_smoothed'] / df['tr_smoothed'])
        
        # Clean up temporary columns
        df.drop(['high_low', 'high_close', 'low_close', 'tr', 'up_move', 'down_move',
                'pos_dm', 'neg_dm', 'tr_smoothed', 'pos_dm_smoothed', 'neg_dm_smoothed'], 
               axis=1, inplace=True)
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        df['stoch_signal'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_pct'] = bb.bollinger_pband()
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # Pivot Points
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['pivot'] = typical_price.rolling(window=20).mean()
        
        # Ensure all indicators are numeric
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        return df
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating technical indicators: {str(e)}"
        )

def prepare_ml_features(df, strategy_name):
    features = {}
    
    if strategy_name == "RSI + MACD":
        features = {
            'rsi': df['rsi'],
            'macd': df['macd'],
            'macd_signal': df['macd_signal'],
            'volume': df['volume'],
            'close': df['close']
        }
    elif strategy_name == "Bollinger + Stochastic":
        features = {
            'bb_high': df['bb_high'],
            'bb_low': df['bb_low'],
            'stoch': df['stoch'],
            'stoch_signal': df['stoch_signal'],
            'volume': df['volume']
        }
    elif strategy_name == "ADX + DMI":
        features = {
            'adx': df['adx'],
            'dmi_plus': df['dmi_plus'],
            'dmi_minus': df['dmi_minus'],
            'volume': df['volume'],
            'close': df['close']
        }
    elif strategy_name == "Volume + MFI":
        features = {
            'volume': df['volume'],
            'volume_sma': df['volume_sma'],
            'mfi': df['mfi'],
            'close': df['close'],
            'volume_ema': df['volume_ema']
        }
    elif strategy_name == "Triple Screen":
        features = {
            'ema_200': df['ema_200'],
            'rsi': df['rsi'],
            'macd': df['macd'],
            'macd_signal': df['macd_signal'],
            'volume': df['volume']
        }
    elif strategy_name == "Trend Following":
        features = {
            'sma_20': df['sma_20'],
            'sma_50': df['sma_50'],
            'adx': df['adx'],
            'bb_high': df['bb_high'],
            'bb_low': df['bb_low']
        }
    elif strategy_name == "Momentum + Volume":
        features = {
            'rsi': df['rsi'],
            'volume': df['volume'],
            'volume_sma': df['volume_sma'],
            'mfi': df['mfi'],
            'close': df['close']
        }
    elif strategy_name == "Support Resistance":
        features = {
            'pivot': df['pivot'],
            'stoch': df['stoch'],
            'cci': df['cci'],
            'close': df['close'],
            'volume': df['volume']
        }
    elif strategy_name == "Volatility Breakout":
        features = {
            'bb_high': df['bb_high'],
            'bb_low': df['bb_low'],
            'atr': df['atr'],
            'volume': df['volume'],
            'volume_sma': df['volume_sma']
        }
    elif strategy_name == "Price Action":
        features = {
            'open': df['open'],
            'close': df['close'],
            'volume': df['volume'],
            'volume_sma': df['volume_sma'],
            'ema_200': df['ema_200']
        }
    
    return pd.DataFrame(features)

def create_ml_models():
    models = {
        "RSI + MACD": RandomForestClassifier(n_estimators=100, random_state=42),
        "Bollinger + Stochastic": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "ADX + DMI": XGBClassifier(n_estimators=100, random_state=42),
        "Volume + MFI": LGBMClassifier(n_estimators=100, random_state=42),
        "Triple Screen": RandomForestClassifier(n_estimators=150, random_state=42),
        "Trend Following": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "Momentum + Volume": XGBClassifier(n_estimators=150, random_state=42),
        "Support Resistance": SVC(probability=True, random_state=42),
        "Volatility Breakout": RandomForestClassifier(n_estimators=200, random_state=42),
        "Price Action": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }
    return models

def train_strategy_models(df):
    models = create_ml_models()
    trained_models = {}
    
    # Create labels (1 for price increase, 0 for decrease)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    for strategy_name, model in models.items():
        try:
            # Prepare features for this strategy
            X = prepare_ml_features(df, strategy_name)
            y = df['target']
            
            # Remove any NaN values
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]
            
            # Train model
            model.fit(X, y)
            trained_models[strategy_name] = model
            
        except Exception as e:
            logger.error(f"Error training model for {strategy_name}: {str(e)}")
            trained_models[strategy_name] = None
    
    return trained_models

def get_ml_predictions(df, trained_models):
    ml_predictions = {}
    
    for strategy_name, model in trained_models.items():
        try:
            if model is not None:
                # Prepare features for this strategy
                X = prepare_ml_features(df, strategy_name)
                
                # Get last row for prediction
                X_last = X.iloc[-1:]
                
                # Make prediction and get probability
                pred_proba = model.predict_proba(X_last)[0]
                prediction = model.predict(X_last)[0]
                
                ml_predictions[strategy_name] = {
                    "prediction": "bullish" if prediction == 1 else "bearish",
                    "confidence": float(max(pred_proba)),
                    "upward_probability": float(pred_proba[1]),
                    "downward_probability": float(pred_proba[0])
                }
            else:
                ml_predictions[strategy_name] = None
                
        except Exception as e:
            logger.error(f"Error getting ML prediction for {strategy_name}: {str(e)}")
            ml_predictions[strategy_name] = None
    
    return ml_predictions

def evaluate_strategies(df):
    strategies = {
        "RSI + MACD": {
            "bullish": (
                (df['rsi'] < 30) & 
                (df['macd'] > df['macd_signal'])
            ),
            "bearish": (
                (df['rsi'] > 70) & 
                (df['macd'] < df['macd_signal'])
            )
        },
        "Bollinger + Stochastic": {
            "bullish": (
                (df['close'] < df['bb_low']) & 
                (df['stoch'] < 20) & 
                (df['stoch'] > df['stoch_signal'])
            ),
            "bearish": (
                (df['close'] > df['bb_high']) & 
                (df['stoch'] > 80) & 
                (df['stoch'] < df['stoch_signal'])
            )
        },
        "ADX + DMI": {
            "bullish": (
                (df['adx'] > 25) & 
                (df['dmi_plus'] > df['dmi_minus'])
            ),
            "bearish": (
                (df['adx'] > 25) & 
                (df['dmi_plus'] < df['dmi_minus'])
            )
        },
        "Volume + MFI": {
            "bullish": (
                (df['volume'] > df['volume_sma']) & 
                (df['mfi'] < 20)
            ),
            "bearish": (
                (df['volume'] > df['volume_sma']) & 
                (df['mfi'] > 80)
            )
        },
        "Triple Screen": {
            "bullish": (
                (df['ema_200'] < df['close']) &  # Long-term trend
                (df['rsi'] < 30) &               # Oversold
                (df['macd'] > df['macd_signal']) # Momentum
            ),
            "bearish": (
                (df['ema_200'] > df['close']) &  # Long-term trend
                (df['rsi'] > 70) &               # Overbought
                (df['macd'] < df['macd_signal']) # Momentum
            )
        },
        "Trend Following": {
            "bullish": (
                (df['sma_20'] > df['sma_50']) &
                (df['adx'] > 25) &
                (df['close'] > df['bb_high'])
            ),
            "bearish": (
                (df['sma_20'] < df['sma_50']) &
                (df['adx'] > 25) &
                (df['close'] < df['bb_low'])
            )
        },
        "Momentum + Volume": {
            "bullish": (
                (df['rsi'] > 50) &
                (df['volume'] > df['volume_sma']) &
                (df['mfi'] > 50)
            ),
            "bearish": (
                (df['rsi'] < 50) &
                (df['volume'] > df['volume_sma']) &
                (df['mfi'] < 50)
            )
        },
        "Support Resistance": {
            "bullish": (
                (df['close'] > df['pivot']) &
                (df['stoch'] < 30) &
                (df['cci'] < -100)
            ),
            "bearish": (
                (df['close'] < df['pivot']) &
                (df['stoch'] > 70) &
                (df['cci'] > 100)
            )
        },
        "Volatility Breakout": {
            "bullish": (
                (df['close'] > df['bb_high']) &
                (df['atr'] > df['atr'].rolling(20).mean()) &
                (df['volume'] > df['volume_sma'])
            ),
            "bearish": (
                (df['close'] < df['bb_low']) &
                (df['atr'] > df['atr'].rolling(20).mean()) &
                (df['volume'] > df['volume_sma'])
            )
        },
        "Price Action": {
            "bullish": (
                (df['close'] > df['open']) &
                (df['close'] > df['close'].shift(1)) &
                (df['volume'] > df['volume_sma'])
            ),
            "bearish": (
                (df['close'] < df['open']) &
                (df['close'] < df['close'].shift(1)) &
                (df['volume'] > df['volume_sma'])
            )
        }
    }

    # Train ML models
    trained_models = train_strategy_models(df)
    
    # Get ML predictions
    ml_predictions = get_ml_predictions(df, trained_models)
    
    # Calculate strategy signals
    last_row = df.iloc[-1]
    strategy_results = {}
    
    for name, conditions in strategies.items():
        bullish = conditions["bullish"].iloc[-1]
        bearish = conditions["bearish"].iloc[-1]
        ml_pred = ml_predictions.get(name)
        
        if ml_pred is not None:
            # Combine traditional signals with ML predictions
            if bullish and ml_pred["prediction"] == "bullish":
                signal = "strong_bullish"
            elif bearish and ml_pred["prediction"] == "bearish":
                signal = "strong_bearish"
            elif bullish or ml_pred["prediction"] == "bullish":
                signal = "bullish"
            elif bearish or ml_pred["prediction"] == "bearish":
                signal = "bearish"
            else:
                signal = "neutral"
                
            strategy_results[name] = {
                "signal": signal,
                "ml_confidence": ml_pred["confidence"],
                "ml_upward_prob": ml_pred["upward_probability"],
                "ml_downward_prob": ml_pred["downward_probability"]
            }
        else:
            # Fallback to traditional signals if ML fails
            if bullish and not bearish:
                signal = "bullish"
            elif bearish and not bullish:
                signal = "bearish"
            else:
                signal = "neutral"
            
            strategy_results[name] = {
                "signal": signal,
                "ml_confidence": None,
                "ml_upward_prob": None,
                "ml_downward_prob": None
            }
    
    # Calculate overall market sentiment
    signal_weights = {
        "strong_bullish": 2,
        "bullish": 1,
        "neutral": 0,
        "bearish": -1,
        "strong_bearish": -2
    }
    
    total_weight = sum(signal_weights[result["signal"]] for result in strategy_results.values())
    max_possible_weight = 2 * len(strategies)
    
    sentiment_score = total_weight / max_possible_weight
    
    if sentiment_score > 0.6:
        overall_sentiment = "strong_uptrend"
    elif sentiment_score > 0.2:
        overall_sentiment = "uptrend"
    elif sentiment_score < -0.6:
        overall_sentiment = "strong_downtrend"
    elif sentiment_score < -0.2:
        overall_sentiment = "downtrend"
    else:
        overall_sentiment = "sideways"
    
    # Calculate summary statistics
    signal_counts = {
        "strong_bullish": sum(1 for r in strategy_results.values() if r["signal"] == "strong_bullish"),
        "bullish": sum(1 for r in strategy_results.values() if r["signal"] == "bullish"),
        "neutral": sum(1 for r in strategy_results.values() if r["signal"] == "neutral"),
        "bearish": sum(1 for r in strategy_results.values() if r["signal"] == "bearish"),
        "strong_bearish": sum(1 for r in strategy_results.values() if r["signal"] == "strong_bearish")
    }
    
    return {
        "strategies": strategy_results,
        "overall_sentiment": overall_sentiment,
        "sentiment_score": float(sentiment_score),
        "summary": signal_counts
    }

def determine_market_state(row):
    # Complex market state determination using multiple indicators
    trend_signals = 0
    
    # ADX trend strength
    if row['adx'] > 25:
        if row['close'] > row['sma_20']:
            trend_signals += 1
        else:
            trend_signals -= 1
    
    # RSI signals
    if row['rsi'] > 70:
        trend_signals -= 1
    elif row['rsi'] < 30:
        trend_signals += 1
    
    # Stochastic signals
    if row['stoch'] > 80:
        trend_signals -= 1
    elif row['stoch'] < 20:
        trend_signals += 1
    
    # Volume confirmation
    if row['volume'] > row['volume_sma']:
        if row['close'] > row['sma_20']:
            trend_signals += 1
        else:
            trend_signals -= 1
    
    # Market state determination
    if trend_signals >= 2:
        return 'uptrend'
    elif trend_signals <= -2:
        return 'downtrend'
    else:
        return 'sideways'

@router.post("/predict")
async def predict_future(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Verify file content
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
            
        try:
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding error. Please ensure the file is UTF-8 encoded")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
            
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
            
        # Verify required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
            
        # Verify data is not empty
        if len(df) < 100:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data. Need at least 100 candles for analysis."
            )
            
        # Convert columns to numeric, replacing any non-numeric values with NaN
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Check for non-numeric data
        if df[required_columns].isna().any().any():
            raise HTTPException(
                status_code=400,
                detail="Non-numeric values found in required columns"
            )
            
        # Remove any rows with NaN values
        df = df.dropna(subset=required_columns)
        
        # Verify we still have enough data after cleaning
        if len(df) < 100:
            raise HTTPException(
                status_code=400,
                detail="Insufficient valid data after cleaning. Need at least 100 valid candles."
            )
            
        try:
            # Calculate basic features
            df['returns'] = df['close'].pct_change()
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Calculate advanced indicators
            df = calculate_advanced_indicators(df)
            
            # Handle any NaN values created by indicators
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Prepare features for prediction
            features = [
                'rsi', 'mfi', 'adx', 'cci', 'williams_r', 'bb_pct', 'atr',
                'force_index', 'stoch', 'stoch_signal', 'volume_sma', 'volatility'
            ]
            
            # Verify all features are present
            missing_features = [feat for feat in features if feat not in df.columns]
            if missing_features:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to calculate indicators: {', '.join(missing_features)}"
                )
            
            X = df[features].copy()  # Create a copy to avoid SettingWithCopyWarning
            
            # Verify no infinity values
            if np.isinf(X.values).any():
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(method='ffill').fillna(method='bfill')
            
            # Prepare target for direction prediction (1 for up, 0 for down)
            y_direction = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_direction, test_size=0.2, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models with error handling
            try:
                rf_classifier = RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )
                rf_classifier.fit(X_train_scaled, y_train)
                
                xgb_classifier = XGBClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
                )
                xgb_classifier.fit(X_train_scaled, y_train)
            except Exception as e:
                logger.error(f"Error training models: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error training models: {str(e)}"
                )
            
            # Get last 100 candles for analysis
            last_100_candles = df.tail(100)
            
            # Predict next 15-min movement
            last_data = scaler.transform(X.iloc[-1:])
            rf_prob = rf_classifier.predict_proba(last_data)[0]
            xgb_prob = xgb_classifier.predict_proba(last_data)[0]
            
            # Ensemble probabilities
            ensemble_prob = (rf_prob + xgb_prob) / 2
            
            # Determine market state
            market_state = determine_market_state(df.iloc[-1])
            
            # Calculate volume analysis
            volume_analysis = {
                'avg_volume': float(last_100_candles['volume'].mean()),
                'volume_trend': 'increasing' if last_100_candles['volume'].iloc[-1] > last_100_candles['volume'].mean() else 'decreasing',
                'volume_strength': float(last_100_candles['volume'].iloc[-1] / last_100_candles['volume'].mean()),
            }
            
            # Calculate strategy signals
            strategy_analysis = evaluate_strategies(df)
            
            # Prepare response
            response = {
                "next_15min_prediction": {
                    "direction_probability": {
                        "upward": float(ensemble_prob[1]),
                        "downward": float(ensemble_prob[0])
                    },
                    "market_state": market_state,
                    "confidence_score": float(max(ensemble_prob)),
                },
                "technical_indicators": {
                    "rsi": float(df['rsi'].iloc[-1]),
                    "adx": float(df['adx'].iloc[-1]),
                    "cci": float(df['cci'].iloc[-1]),
                    "stochastic": float(df['stoch'].iloc[-1]),
                    "williams_r": float(df['williams_r'].iloc[-1]),
                    "mfi": float(df['mfi'].iloc[-1])
                },
                "volume_analysis": volume_analysis,
                "market_strength": {
                    "trend_strength": float(df['adx'].iloc[-1]),
                    "volatility": float(df['volatility'].iloc[-1]),
                    "momentum": float(df['rsi'].iloc[-1])
                },
                "strategy_analysis": strategy_analysis
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing data: {str(e)}"
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
