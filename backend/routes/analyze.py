from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from io import StringIO
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import sys
import os
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.indicators import calculate_rsi

router = APIRouter()

def calculate_advanced_features(df):
    """Calculate advanced technical indicators and features"""
    try:
        # Volume-based indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ema'] = df['volume'].ewm(span=20, adjust=False).mean()
        df['force_index'] = ta.volume.force_index(close=df['close'], volume=df['volume'], window=13)
        df['ease_of_movement'] = ta.volume.ease_of_movement(high=df['high'], low=df['low'], volume=df['volume'], window=14)
        df['volume_price_trend'] = (df['volume'] * (df['close'] - df['close'].shift(1))).cumsum()

        # Momentum indicators
        df['awesome_oscillator'] = ta.momentum.awesome_oscillator(high=df['high'], low=df['low'])
        df['kama'] = ta.momentum.kama(close=df['close'], window=10, pow1=2, pow2=30)
        df['ppo'] = ta.momentum.ppo(close=df['close'])
        df['pvo'] = ((df['volume'].ewm(span=12, adjust=False).mean() - df['volume'].ewm(span=26, adjust=False).mean()) / 
                     df['volume'].ewm(span=26, adjust=False).mean() * 100)
        df['roc'] = ta.momentum.roc(close=df['close'], window=12)
        df['stoch_rsi'] = ta.momentum.stochrsi(close=df['close'])

        # Trend indicators
        df['adx'] = ta.trend.adx(high=df['high'], low=df['low'], close=df['close'])
        df['cci'] = ta.trend.cci(high=df['high'], low=df['low'], close=df['close'])
        df['dpo'] = ta.trend.dpo(close=df['close'])
        df['ichimoku_a'] = ta.trend.ichimoku_a(high=df['high'], low=df['low'])
        df['ichimoku_b'] = ta.trend.ichimoku_b(high=df['high'], low=df['low'])
        df['mass_index'] = ta.trend.mass_index(high=df['high'], low=df['low'])
        df['trix'] = ta.trend.ema_indicator(close=df['close'], window=18)  # Using EMA as alternative for TRIX
        df['vortex_pos'] = ta.trend.vortex_indicator_pos(high=df['high'], low=df['low'], close=df['close'])
        df['vortex_neg'] = ta.trend.vortex_indicator_neg(high=df['high'], low=df['low'], close=df['close'])

        # Volatility indicators
        bb = ta.volatility.BollingerBands(close=df['close'])
        df['bbw'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        df['dcw'] = ta.volatility.donchian_channel_wband(high=df['high'], low=df['low'], close=df['close'])
        df['kc_width'] = (ta.volatility.keltner_channel_hband(high=df['high'], low=df['low'], close=df['close']) - 
                         ta.volatility.keltner_channel_lband(high=df['high'], low=df['low'], close=df['close']))
        df['atr'] = ta.volatility.average_true_range(high=df['high'], low=df['low'], close=df['close'])

        # Custom features
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        return df
    except Exception as e:
        print(f"Error in calculate_advanced_features: {str(e)}")
        return df

def apply_trading_strategies(df):
    """Apply various trading strategies on the data"""
    try:
        strategies = {}
        
        # Ensure all required columns exist before using them
        required_columns = ['rsi', 'macd', 'macd_signal', 'bb_lower', 'bb_upper', 'sma_5', 'ema_5',
                          'volume_price_trend', 'adx', 'vortex_pos', 'vortex_neg', 'cci',
                          'ichimoku_a', 'ichimoku_b', 'trix', 'force_index', 'ease_of_movement',
                          'awesome_oscillator', 'stoch_rsi', 'atr', 'roc', 'ppo', 'kama', 'close']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns for strategies: {missing_columns}")
            return pd.DataFrame()  # Return empty DataFrame if missing columns
        
        # 1. RSI Strategy
        strategies['rsi_signal'] = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
        
        # 2. MACD Strategy
        strategies['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # 3. Bollinger Bands Strategy
        strategies['bb_signal'] = np.where(df['close'] < df['bb_lower'], 1, 
                                         np.where(df['close'] > df['bb_upper'], -1, 0))
        
        # 4. Moving Average Cross Strategy
        strategies['ma_cross'] = np.where(df['sma_5'] > df['ema_5'], 1, -1)
        
        # 5. Volume Price Trend Strategy
        strategies['vpt_signal'] = np.where(df['volume_price_trend'] > df['volume_price_trend'].shift(1), 1, -1)
        
        # 6. ADX Strategy
        strategies['adx_signal'] = np.where((df['adx'] > 25) & (df['vortex_pos'] > df['vortex_neg']), 1,
                                          np.where((df['adx'] > 25) & (df['vortex_pos'] < df['vortex_neg']), -1, 0))
        
        # 7. CCI Strategy
        strategies['cci_signal'] = np.where(df['cci'] < -100, 1, np.where(df['cci'] > 100, -1, 0))
        
        # 8. Ichimoku Strategy
        strategies['ichimoku_signal'] = np.where(df['ichimoku_a'] > df['ichimoku_b'], 1, -1)
        
        # 9. TRIX Strategy
        strategies['trix_signal'] = np.where(df['trix'] > df['trix'].shift(1), 1, -1)
        
        # 10. Force Index Strategy
        strategies['force_index_signal'] = np.where(df['force_index'] > 0, 1, -1)
        
        # 11. Ease of Movement Strategy
        strategies['eom_signal'] = np.where(df['ease_of_movement'] > 0, 1, -1)
        
        # 12. Awesome Oscillator Strategy
        strategies['ao_signal'] = np.where(df['awesome_oscillator'] > 0, 1, -1)
        
        # 13. Stochastic RSI Strategy
        strategies['stoch_rsi_signal'] = np.where(df['stoch_rsi'] < 0.2, 1, np.where(df['stoch_rsi'] > 0.8, -1, 0))
        
        # 14. Volatility Breakout Strategy
        strategies['volatility_breakout'] = np.where(df['close'] > df['close'].shift(1) + df['atr'], 1,
                                                   np.where(df['close'] < df['close'].shift(1) - df['atr'], -1, 0))
        
        # 15. Combined Momentum Strategy
        strategies['momentum_combined'] = np.where((df['roc'] > 0) & (df['ppo'] > 0) & (df['kama'] > df['close']), 1, -1)
        
        # 16. KAMA Cross Strategy
        strategies['kama_cross'] = np.where(df['close'] > df['kama'], 1, -1)
        
        # 17. Volume Surge Strategy
        strategies['volume_surge'] = np.where(df['volume'] > df['volume'].rolling(20).mean() * 2, 1, -1)
        
        # 18. Price Channel Breakout Strategy
        strategies['price_channel_breakout'] = np.where(df['close'] > df['high'].rolling(20).max(), 1,
                                                      np.where(df['close'] < df['low'].rolling(20).min(), -1, 0))
        
        # 19. Triple EMA Strategy
        strategies['triple_ema'] = np.where((df['ema_5'] > df['ema_10']) & (df['ema_10'] > df['ema_20']), 1,
                                          np.where((df['ema_5'] < df['ema_10']) & (df['ema_10'] < df['ema_20']), -1, 0))
        
        # 20. Elder Ray Strategy
        strategies['elder_ray'] = np.where((df['high'] > df['ema_13']) & (df['low'] > df['ema_13']), 1,
                                         np.where((df['high'] < df['ema_13']) & (df['low'] < df['ema_13']), -1, 0))
        
        # 21. Supertrend Strategy
        strategies['supertrend'] = np.where(df['close'] > (df['high'].rolling(10).mean() + df['atr'] * 3), 1,
                                          np.where(df['close'] < (df['low'].rolling(10).mean() - df['atr'] * 3), -1, 0))
        
        # 22. Squeeze Momentum Strategy
        strategies['squeeze_momentum'] = np.where((df['bb_upper'] - df['bb_lower']) < df['atr'] * 2, 1, -1)
        
        # 23. VWAP Cross Strategy
        strategies['vwap_cross'] = np.where(df['close'] > df['vwap'], 1, -1)
        
        # 24. DMI Strategy
        strategies['dmi_signal'] = np.where((df['plus_di'] > df['minus_di']) & (df['adx'] > 25), 1,
                                          np.where((df['plus_di'] < df['minus_di']) & (df['adx'] > 25), -1, 0))
        
        # 25. Williams R Strategy
        strategies['williams_r'] = np.where(df['williams_r'] < -80, 1, np.where(df['williams_r'] > -20, -1, 0))
        
        # 26. Ultimate Oscillator Strategy
        strategies['ultimate_oscillator'] = np.where(df['ultimate_oscillator'] < 30, 1, 
                                                   np.where(df['ultimate_oscillator'] > 70, -1, 0))
        
        # 27. Aroon Oscillator Strategy
        strategies['aroon_oscillator'] = np.where(df['aroon_oscillator'] > 50, 1, 
                                                np.where(df['aroon_oscillator'] < -50, -1, 0))
        
        # 28. Chaikin Money Flow Strategy
        strategies['chaikin_money_flow'] = np.where(df['cmf'] > 0.2, 1, np.where(df['cmf'] < -0.2, -1, 0))
        
        # 29. On Balance Volume Strategy
        strategies['on_balance_volume'] = np.where(df['obv'] > df['obv'].shift(1), 1, -1)
        
        # 30. Momentum Quality Strategy
        strategies['momentum_quality'] = np.where((df['rsi'] > 50) & (df['macd'] > 0) & (df['ao'] > 0), 1,
                                                np.where((df['rsi'] < 50) & (df['macd'] < 0) & (df['ao'] < 0), -1, 0))
        
        # 31. Trend Strength Strategy
        strategies['trend_strength'] = np.where((df['adx'] > 25) & (df['plus_di'] > df['minus_di']), 1,
                                              np.where((df['adx'] > 25) & (df['plus_di'] < df['minus_di']), -1, 0))
        
        # 32. Volume Trend Confirm Strategy
        strategies['volume_trend_confirm'] = np.where((df['close'] > df['sma_20']) & (df['volume'] > df['volume_sma']), 1,
                                                    np.where((df['close'] < df['sma_20']) & (df['volume'] > df['volume_sma']), -1, 0))
        
        # 33. Volatility Regime Strategy
        strategies['volatility_regime'] = np.where(df['atr'] < df['atr'].rolling(20).mean(), 1, -1)
        
        # 34. Price Momentum Strategy
        strategies['price_momentum'] = np.where((df['close'] - df['close'].shift(10))/df['close'].shift(10) > 0.02, 1,
                                              np.where((df['close'] - df['close'].shift(10))/df['close'].shift(10) < -0.02, -1, 0))
        
        # 35. Trend Reversal Strategy
        strategies['trend_reversal'] = np.where((df['macd'] > 0) & (df['rsi'] < 30) & (df['cci'] < -100), 1,
                                              np.where((df['macd'] < 0) & (df['rsi'] > 70) & (df['cci'] > 100), -1, 0))
        
        # Calculate Combined Signal
        strategies_df = pd.DataFrame(strategies)
        combined_signal = strategies_df.mean(axis=1)
        
        # Determine Overall Market Direction
        threshold = 0.2  # Adjustable threshold for signal strength
        market_direction = np.where(combined_signal > threshold, 'Uptrend',
                                  np.where(combined_signal < -threshold, 'Downtrend', 'Sideways'))
        
        strategies['combined_signal'] = combined_signal
        strategies['market_direction'] = market_direction
        
        return pd.DataFrame(strategies)
    except Exception as e:
        print(f"Error in apply_trading_strategies: {str(e)}")
        return pd.DataFrame()

def predict_market_direction(df, strategies_df):
    """Predict market direction for next 15 minutes using ensemble of models"""
    try:
        # Prepare features for last 100 candles
        df_last_100 = df.tail(100).copy()
        
        # Feature selection - only use features that exist
        all_features = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_5', 'ema_5', 'volatility',
                       'volume_sma', 'volume_ema', 'force_index', 'ease_of_movement', 'volume_price_trend',
                       'awesome_oscillator', 'kama', 'ppo', 'pvo', 'roc', 'stoch_rsi', 'adx', 'cci',
                       'dpo', 'mass_index', 'trix', 'vortex_pos', 'vortex_neg', 'bbw', 'dcw', 'kc_width',
                       'atr', 'hlc3', 'ohlc4', 'high_low_ratio', 'close_open_ratio']
        
        feature_columns = [col for col in all_features if col in df_last_100.columns]
        if not feature_columns:
            raise ValueError("No valid features found for prediction")
        
        X = df_last_100[feature_columns].fillna(method='ffill').fillna(method='bfill')
        
        # Create target variable (1: uptrend, 0: sideways, -1: downtrend)
        future_returns = df_last_100['close'].pct_change(3).shift(-3)
        y = np.where(future_returns > 0.001, 1, np.where(future_returns < -0.001, -1, 0))
        y = y[:-3]  # Remove last 3 NaN values
        X = X[:-3]  # Align with target
        
        if len(X) < 2:  # Need at least 2 samples for train/test split
            raise ValueError("Insufficient data points for prediction")
        
        # Split data
        X_train, X_test = X[:-1], X.iloc[[-1]]
        y_train = y[:-1]
        
        # Initialize models with reduced complexity for small datasets
        models = {
            'rf': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
            'xgb': XGBClassifier(n_estimators=50, max_depth=3, random_state=42),
            'catboost': CatBoostClassifier(iterations=50, depth=3, random_state=42, verbose=False),
            'lgbm': LGBMClassifier(n_estimators=50, max_depth=3, random_state=42)
        }
        
        # Train models and make predictions
        predictions = {}
        probabilities = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_test)[0]
            probabilities[name] = model.predict_proba(X_test)[0]
        
        # Calculate ensemble prediction
        ensemble_pred = np.sign(sum(predictions.values()))
        
        # Calculate confidence scores
        confidence_scores = {
            'rf': float(max(probabilities['rf'])),
            'xgb': float(max(probabilities['xgb'])),
            'catboost': float(max(probabilities['catboost'])),
            'lgbm': float(max(probabilities['lgbm']))
        }
        
        # Get strategy signals for current candle
        current_signals = strategies_df.iloc[-1].to_dict() if not strategies_df.empty else {}
        
        # Determine market condition
        market_condition = {
            1: "Uptrend",
            0: "Sideways",
            -1: "Downtrend"
        }[ensemble_pred]
        
        return {
            "market_direction": market_condition,
            "confidence_scores": confidence_scores,
            "strategy_signals": current_signals,
            "prediction_metrics": {
                "model_agreement": len([p for p in predictions.values() if p == ensemble_pred]) / len(predictions),
                "average_confidence": float(np.mean(list(confidence_scores.values()))),
                "volatility_level": float(df_last_100['atr'].iloc[-1] / df_last_100['close'].iloc[-1]) if 'atr' in df_last_100.columns else None,
                "trend_strength": float(df_last_100['adx'].iloc[-1]) if 'adx' in df_last_100.columns else None
            }
        }
    except Exception as e:
        print(f"Error in predict_market_direction: {str(e)}")
        return {
            "market_direction": "Unknown",
            "error": str(e)
        }

def get_strategy_description(strategy_name):
    """Return description for each trading strategy"""
    descriptions = {
        'rsi_signal': 'RSI-based strategy: Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought)',
        'macd_signal': 'MACD strategy: Buy when MACD crosses above signal line, Sell when below',
        'bb_signal': 'Bollinger Bands strategy: Buy at lower band, Sell at upper band',
        'ma_cross': 'Moving Average Cross strategy: Buy when short MA crosses above long MA, Sell when below',
        'vpt_signal': 'Volume Price Trend strategy: Buy on positive trend, Sell on negative trend',
        'adx_signal': 'ADX Strategy: Buy/Sell based on trend strength and Vortex indicators',
        'cci_signal': 'CCI Strategy: Buy when CCI < -100 (oversold), Sell when CCI > 100 (overbought)',
        'ichimoku_signal': 'Ichimoku Cloud strategy: Buy above cloud, Sell below cloud',
        'trix_signal': 'TRIX strategy: Buy on positive crossover, Sell on negative crossover',
        'force_index_signal': 'Force Index strategy: Buy on positive force, Sell on negative force',
        'eom_signal': 'Ease of Movement strategy: Buy on positive EOM, Sell on negative EOM',
        'ao_signal': 'Awesome Oscillator strategy: Buy on positive AO, Sell on negative AO',
        'stoch_rsi_signal': 'Stochastic RSI strategy: Buy < 0.2 (oversold), Sell > 0.8 (overbought)',
        'volatility_breakout': 'Volatility Breakout strategy: Buy/Sell on ATR-based breakouts',
        'momentum_combined': 'Combined Momentum strategy using ROC, PPO, and KAMA',
        'kama_cross': 'KAMA Cross strategy: Buy above KAMA, Sell below KAMA',
        'volume_surge': 'Volume Surge strategy: Signals based on abnormal volume',
        'price_channel_breakout': 'Price Channel Breakout strategy using highs/lows',
        'triple_ema': 'Triple EMA strategy using 5, 10, and 20-period EMAs',
        'elder_ray': 'Elder Ray strategy using EMA and high/low prices',
        'supertrend': 'Supertrend strategy using ATR-based trend following',
        'squeeze_momentum': 'Squeeze Momentum strategy using Bollinger Bands and ATR',
        'vwap_cross': 'VWAP Cross strategy: Buy above VWAP, Sell below VWAP',
        'dmi_signal': 'DMI strategy using ADX and DI indicators'
    }
    return descriptions.get(strategy_name, 'Strategy description not available')

@router.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    try:
        print(f"Starting analysis for file: {file.filename}")
        
        # Read the contents of the file
        contents = await file.read()
        print("File contents read successfully")
        
        # Create a StringIO object from the contents
        from io import StringIO
        import io
        
        # Try to decode the contents as UTF-8
        try:
            str_io = StringIO(contents.decode('utf-8'))
            print("File decoded as UTF-8 successfully")
        except UnicodeDecodeError as e:
            print(f"UTF-8 decode error: {e}")
            str_io = io.BytesIO(contents)
            print("Falling back to BytesIO")
        
        # Read CSV file
        try:
            df = pd.read_csv(str_io)
            print(f"CSV read successfully. Shape: {df.shape}")
            print(f"Columns found: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} to numeric. Null values: {df[col].isnull().sum()}")
            except Exception as e:
                print(f"Error converting {col} to numeric: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error converting {col} to numeric values: {str(e)}"
                )
            
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                if df['timestamp'].max() > 1e10:  # Convert from milliseconds to seconds if needed
                    df['timestamp'] = df['timestamp'] / 1000
                print("Timestamp conversion successful")
            except Exception as e:
                print(f"Error converting timestamp: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing timestamp: {str(e)}"
                )
        
        # Add datetime column if not present
        if 'datetime' not in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                print("Created datetime column from timestamp")
            except Exception as e:
                print(f"Error creating datetime column: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error creating datetime column: {str(e)}"
                )
        
        # Calculate technical indicators
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # Calculate MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Calculate Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # Calculate Moving Averages
            df['sma_5'] = ta.trend.SMAIndicator(df['close'], window=5).sma_indicator()
            df['ema_5'] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
            
            # Calculate Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            print("Technical indicators calculated successfully")
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error calculating technical indicators: {str(e)}"
            )
        
        # Calculate advanced features
        df = calculate_advanced_features(df)
        
        # Apply trading strategies
        strategies_df = apply_trading_strategies(df)
        
        # Predict market direction
        market_direction = predict_market_direction(df, strategies_df)
        
        # Prepare features for ML models
        try:
            features = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_5', 'ema_5', 'volatility']
            X = df[features].fillna(method='ffill').fillna(method='bfill')
            y = df['returns'].shift(-1).fillna(method='ffill')  # Predict next period's returns
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            print(f"Data split complete. Training size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print("Feature scaling complete")
            
        except Exception as e:
            print(f"Error preparing ML features: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error preparing machine learning features: {str(e)}"
            )
        
        # Train models with appropriate parameters for small datasets
        try:
            # Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_split=2,
                random_state=42
            )
            
            # XGBoost model
            xgb_model = XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            
            # Train models
            rf_model.fit(X_train_scaled, y_train)
            xgb_model.fit(X_train_scaled, y_train)
            print("Models trained successfully")
            
            # Make predictions
            rf_pred = rf_model.predict(X_test_scaled)
            xgb_pred = xgb_model.predict(X_test_scaled)
            print("Predictions generated")
            
        except Exception as e:
            print(f"Error in model training/prediction: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error training ML models: {str(e)}"
            )
        
        # Prepare response
        try:
            # Ensure datetime is properly formatted
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            analysis_results = {
                "technical_indicators": df[['datetime', 'close', 'macd', 'rsi', 'bb_upper', 'bb_lower', 'sma_5', 'ema_5']].apply(lambda x: {
                    'datetime': x['datetime'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x['datetime']) else None,
                    'close': float(x['close']) if pd.notnull(x['close']) else None,
                    'macd': float(x['macd']) if pd.notnull(x['macd']) else None,
                    'rsi': float(x['rsi']) if pd.notnull(x['rsi']) else None,
                    'bb_upper': float(x['bb_upper']) if pd.notnull(x['bb_upper']) else None,
                    'bb_lower': float(x['bb_lower']) if pd.notnull(x['bb_lower']) else None,
                    'sma_5': float(x['sma_5']) if pd.notnull(x['sma_5']) else None,
                    'ema_5': float(x['ema_5']) if pd.notnull(x['ema_5']) else None,
                }, axis=1).tolist(),
                "trading_signals": {
                    name: {
                        "signal": int(signal.iloc[-1]) if pd.notnull(signal.iloc[-1]) else 0,
                        "description": get_strategy_description(name)
                    } for name, signal in strategies_df.items()
                },
                "predictions": {
                    "random_forest": rf_pred.tolist(),
                    "xgboost": xgb_pred.tolist(),
                    "metadata": {
                        "data_points": len(df),
                        "training_size": len(X_train),
                        "test_size": len(X_test)
                    },
                    "future": {
                        "dates": [
                            (df['datetime'].iloc[-1] + pd.Timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S')
                            for i in range(1, 11)  # Next 10 days
                        ],
                        "ensemble_predictions": [
                            float(df['close'].iloc[-1] * (1 + (rf_pred[-1] + xgb_pred[-1]) / 2) ** i)
                            for i in range(1, 11)
                        ],
                        "model_predictions": {
                            "random_forest": [
                                float(df['close'].iloc[-1] * (1 + rf_pred[-1]) ** i)
                                for i in range(1, 11)
                            ],
                            "xgboost": [
                                float(df['close'].iloc[-1] * (1 + xgb_pred[-1]) ** i)
                                for i in range(1, 11)
                            ]
                        },
                        "upper_bound": [
                            float(df['close'].iloc[-1] * (1 + (rf_pred[-1] + xgb_pred[-1]) / 2 + df['volatility'].iloc[-1]) ** i)
                            for i in range(1, 11)
                        ],
                        "lower_bound": [
                            float(df['close'].iloc[-1] * (1 + (rf_pred[-1] + xgb_pred[-1]) / 2 - df['volatility'].iloc[-1]) ** i)
                            for i in range(1, 11)
                        ]
                    }
                },
                "summary_stats": {
                    "current_price": float(df['close'].iloc[-1]) if pd.notnull(df['close'].iloc[-1]) else None,
                    "price_change": float(df['returns'].iloc[-1] * 100) if pd.notnull(df['returns'].iloc[-1]) else None,
                    "current_rsi": float(df['rsi'].iloc[-1]) if pd.notnull(df['rsi'].iloc[-1]) else None,
                    "current_volatility": float(df['volatility'].iloc[-1] * 100) if pd.notnull(df['volatility'].iloc[-1]) else None
                },
                "advanced_analysis": market_direction
            }
            print("Analysis results prepared successfully")
            return analysis_results
            
        except Exception as e:
            print(f"Error preparing response: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error preparing analysis results: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
