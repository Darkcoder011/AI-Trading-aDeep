from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
import os
os.environ['TF_KERAS'] = '1'  # Force using tf-keras compatibility mode
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class MarketAnalysisRequest(BaseModel):
    model_name: str
    csv_data: str  # Base64 encoded CSV data

class MarketPrediction(BaseModel):
    trend: str
    confidence: float
    timeframe: str
    volume_analysis: Dict[str, Any]
    pressure_analysis: Dict[str, Any]
    support_resistance: Dict[str, float]

router = APIRouter()

# Define available LLM models
LLM_MODELS = {
    "finbert": {
        "name": "ProsusAI/finbert",
        "type": "sentiment",
        "description": "Specialized in financial sentiment analysis"
    },
    "bert_market": {
        "name": "bert-base-market",
        "type": "market_analysis",
        "description": "Fine-tuned BERT for pattern recognition"
    },
    "gpt2_trading": {
        "name": "gpt2-trading",
        "type": "prediction",
        "description": "GPT-2 fine-tuned on trading patterns"
    },
    "roberta_market": {
        "name": "roberta-base-market",
        "type": "technical_analysis",
        "description": "RoBERTa for technical analysis"
    },
    "xlnet_market": {
        "name": "xlnet-market",
        "type": "volume_analysis",
        "description": "XLNet for volume and price action analysis"
    },
    "distilbert_market": {
        "name": "distilbert-market",
        "type": "momentum_analysis",
        "description": "DistilBERT for momentum analysis"
    },
    "albert_market": {
        "name": "albert-market",
        "type": "support_resistance",
        "description": "ALBERT for support/resistance analysis"
    },
    "electra_market": {
        "name": "electra-market",
        "type": "trend_analysis",
        "description": "ELECTRA for trend strength analysis"
    },
    "longformer_market": {
        "name": "longformer-market",
        "type": "long_term_patterns",
        "description": "Longformer for long-term pattern analysis"
    },
    "deberta_market": {
        "name": "deberta-market",
        "type": "market_sentiment",
        "description": "DeBERTa for advanced market sentiment"
    }
}

def calculate_market_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key market metrics from candlestick data"""
    # Calculate volume metrics
    volume_sma = df["volume"].rolling(window=20).mean()
    volume_std = df["volume"].rolling(window=20).std()
    
    # Calculate price action metrics
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]
    
    # Calculate buying/selling pressure
    buying_pressure = df[df["close"] > df["open"]]["volume"].sum()
    selling_pressure = df[df["close"] < df["open"]]["volume"].sum()
    total_pressure = buying_pressure + selling_pressure
    
    # Calculate advanced metrics
    price_momentum = df["close"].pct_change().rolling(window=14).mean()
    rsi = calculate_rsi(df["close"])
    
    metrics = {
        "volume_analysis": {
            "average_volume": float(df["volume"].mean()),
            "volume_trend": "increasing" if df["volume"].pct_change().mean() > 0 else "decreasing",
            "volume_strength": float(df["volume"].tail(20).mean() / df["volume"].mean()),
            "volume_volatility": float(volume_std.iloc[-1] / volume_sma.iloc[-1])
        },
        "pressure_analysis": {
            "buying_pressure": float(buying_pressure / total_pressure),
            "selling_pressure": float(selling_pressure / total_pressure),
            "pressure_ratio": float((buying_pressure - selling_pressure) / total_pressure),
            "buying_volume_avg": float(df[df["close"] > df["open"]]["volume"].mean()),
            "selling_volume_avg": float(df[df["close"] < df["open"]]["volume"].mean()),
            "net_money_flow": float(money_flow.sum()),
            "pressure_momentum": float(price_momentum.iloc[-1]),
            "rsi": float(rsi.iloc[-1])
        },
        "trend_metrics": {
            "price_momentum": float(df["close"].pct_change().mean()),
            "volatility": float(df["high"].div(df["low"]).std()),
            "trend_strength": float(df["close"].tail(20).mean() / df["close"].mean()),
            "price_velocity": float(df["close"].diff().mean()),
            "price_acceleration": float(df["close"].diff().diff().mean())
        }
    }
    return metrics

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_market_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze market structure and patterns"""
    structure = {
        "support_levels": [
            float(df["low"].rolling(20).min().iloc[-1]),
            float(df["low"].rolling(50).min().iloc[-1])
        ],
        "resistance_levels": [
            float(df["high"].rolling(20).max().iloc[-1]),
            float(df["high"].rolling(50).max().iloc[-1])
        ],
        "market_phase": determine_market_phase(df),
        "pattern_probability": identify_patterns(df)
    }
    return structure

def determine_market_phase(df: pd.DataFrame) -> str:
    """Determine the current market phase"""
    sma20 = df["close"].rolling(20).mean()
    sma50 = df["close"].rolling(50).mean()
    current_price = df["close"].iloc[-1]
    
    if current_price > sma20.iloc[-1] > sma50.iloc[-1]:
        return "strong_uptrend"
    elif current_price < sma20.iloc[-1] < sma50.iloc[-1]:
        return "strong_downtrend"
    elif sma20.iloc[-1] > sma50.iloc[-1] and current_price < sma20.iloc[-1]:
        return "weak_uptrend"
    elif sma20.iloc[-1] < sma50.iloc[-1] and current_price > sma20.iloc[-1]:
        return "weak_downtrend"
    else:
        return "sideways"

def identify_patterns(df: pd.DataFrame) -> Dict[str, float]:
    """Identify common chart patterns and their probabilities"""
    patterns = {
        "double_top": 0.0,
        "double_bottom": 0.0,
        "head_shoulders": 0.0,
        "triangle": 0.0
    }
    # Add pattern recognition logic here
    return patterns

@router.post("/llm-analysis/predict")
async def analyze_market(file: UploadFile = File(...), model_name: str = "finbert") -> Dict[str, Any]:
    """
    Analyze market data using selected LLM model and provide predictions
    """
    try:
        if model_name not in LLM_MODELS:
            raise HTTPException(status_code=400, detail="Invalid model name")

        # Read and process CSV data
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Calculate market metrics
        metrics = calculate_market_metrics(df)
        structure = analyze_market_structure(df)
        
        # Prepare market data description for LLM
        market_description = generate_market_description(df, metrics, structure)
        
        # Get LLM prediction
        prediction = get_llm_prediction(market_description, model_name)
        
        # Construct a consistent response
        return {
            "model_name": model_name,
            "trend": prediction.get('trend', 'Neutral'),
            "confidence": prediction.get('confidence', 0.5),
            "timeframe": prediction.get('timeframe', 'Short-term'),
            
            "volume_analysis": metrics.get('volume_analysis', {
                "average_volume": 0,
                "volume_trend": "Neutral",
                "volume_strength": 0,
                "volume_volatility": 0
            }),
            
            "pressure_analysis": metrics.get('pressure_analysis', {
                "buying_pressure": 0,
                "selling_pressure": 0,
                "pressure_ratio": 0,
                "buying_volume_avg": 0,
                "selling_volume_avg": 0,
                "net_money_flow": 0,
                "pressure_momentum": 0,
                "rsi": 0
            }),
            
            "support_resistance": {
                "support_levels": structure.get('support_levels', []),
                "resistance_levels": structure.get('resistance_levels', [])
            },
            
            "market_structure": {
                "market_phase": structure.get('market_phase', 'Undefined'),
                "support_levels": structure.get('support_levels', []),
                "resistance_levels": structure.get('resistance_levels', []),
                "pattern_probability": structure.get('pattern_probability', {})
            },
            
            # Add predictions from different models if available
            "predictions": {
                model_name: prediction
                for model_name, prediction in get_model_predictions(market_description).items()
            } if 'get_model_predictions' in globals() else {}
        }
    except Exception as e:
        logging.error(f"Error in market analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during market analysis: {str(e)}")

def generate_market_description(df: pd.DataFrame, metrics: Dict[str, Any], structure: Dict[str, Any]) -> str:
    """Generate a textual description of market conditions for LLM input"""
    latest_price = df["close"].iloc[-1]
    price_change = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    
    description = f"""
    Market Analysis for the last 100 candles:
    Current Price: {latest_price:.2f}
    Price Change: {price_change:.2f}%
    Volume Trend: {metrics['volume_analysis']['volume_trend']}
    Market Phase: {structure['market_phase']}
    Buying Pressure: {metrics['pressure_analysis']['buying_pressure']:.2f}
    Selling Pressure: {metrics['pressure_analysis']['selling_pressure']:.2f}
    """
    return description

def get_llm_prediction(market_description: str, model_name: str) -> Dict[str, Any]:
    """Get prediction from specified LLM model for next 15 minutes"""
    try:
        model_info = LLM_MODELS[model_name]
        model = AutoModelForSequenceClassification.from_pretrained(model_info["name"])
        tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
        
        # Prepare prompt for 15-minute prediction
        prompt = f"""Analyze the following market data for next 15 minutes prediction:
        {market_description}
        Focus on: Price action, Volume patterns, Market structure, Support/Resistance levels
        Timeframe: Next 15 minutes
        """
        
        # Get model prediction
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        
        # Ensure we have the correct number of labels
        num_labels = outputs.logits.shape[1]
        predictions = torch.softmax(outputs.logits, dim=1)
        confidence_scores = predictions[0].tolist()
        
        # Define directions based on number of labels
        if num_labels == 5:
            directions = ["strong_uptrend", "weak_uptrend", "sideways", "weak_downtrend", "strong_downtrend"]
        elif num_labels == 3:
            directions = ["uptrend", "sideways", "downtrend"]
        else:
            # Fallback for any other number of labels
            directions = [f"class_{i}" for i in range(num_labels)]
        
        # Safely get the highest confidence prediction
        max_confidence_idx = confidence_scores.index(max(confidence_scores))
        if max_confidence_idx >= len(directions):
            max_confidence_idx = 0  # Fallback to first class if index is out of range
            
        predicted_trend = directions[max_confidence_idx]
        confidence = confidence_scores[max_confidence_idx]
        
        # Calculate strength based on confidence
        strength = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        
        # Safely create direction probabilities
        direction_probabilities = {}
        for i, score in enumerate(confidence_scores):
            if i < len(directions):
                direction_probabilities[directions[i]] = round(score, 4)
            else:
                break
        
        return {
            "model_name": model_name,
            "prediction": {
                "trend": predicted_trend,
                "confidence": confidence,
                "strength": strength,
                "timeframe": "15min",
                "timestamp": datetime.now().isoformat(),
                "direction_probabilities": direction_probabilities
            },
            "type": model_info["type"],
            "name": model_info["name"],
            "description": model_info["description"]
        }
    except Exception as e:
        logging.error(f"Error in model prediction: {str(e)}")
        return {
            "model_name": model_name,
            "prediction": None,
            "type": LLM_MODELS[model_name]["type"],
            "name": LLM_MODELS[model_name]["name"],
            "description": LLM_MODELS[model_name]["description"],
            "error": str(e)
        }

def calculate_consensus_prediction(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate consensus from all model predictions for 15-min forecast"""
    valid_predictions = [
        p for p in predictions.values() 
        if p.get("prediction") and p["prediction"].get("trend")
    ]
    
    if not valid_predictions:
        return None
        
    # Collect all trend predictions and their confidences
    trend_votes = {
        "strong_uptrend": 0,
        "weak_uptrend": 0,
        "sideways": 0,
        "weak_downtrend": 0,
        "strong_downtrend": 0
    }
    
    weighted_confidence = {trend: 0.0 for trend in trend_votes.keys()}
    
    # Calculate weighted votes
    for pred in valid_predictions:
        trend = pred["prediction"]["trend"]
        confidence = pred["prediction"]["confidence"]
        trend_votes[trend] += 1
        weighted_confidence[trend] += confidence
    
    # Find dominant trend
    max_votes = max(trend_votes.values())
    top_trends = [
        trend for trend, votes in trend_votes.items() 
        if votes == max_votes
    ]
    
    # If multiple trends have same votes, use confidence as tiebreaker
    if len(top_trends) > 1:
        dominant_trend = max(
            top_trends,
            key=lambda t: weighted_confidence[t]
        )
    else:
        dominant_trend = top_trends[0]
    
    # Calculate agreement level
    total_predictions = len(valid_predictions)
    agreement_percentage = trend_votes[dominant_trend] / total_predictions
    
    agreement_level = (
        "strong" if agreement_percentage >= 0.7
        else "moderate" if agreement_percentage >= 0.5
        else "weak"
    )
    
    # Calculate average confidence for dominant trend
    avg_confidence = (
        weighted_confidence[dominant_trend] / trend_votes[dominant_trend]
        if trend_votes[dominant_trend] > 0
        else 0.0
    )
    
    return {
        "trend": dominant_trend,
        "confidence": avg_confidence,
        "agreement_level": agreement_level,
        "timeframe": "15min",
        "timestamp": datetime.now().isoformat(),
        "model_distribution": {
            trend: {
                "votes": votes,
                "percentage": votes/total_predictions,
                "avg_confidence": weighted_confidence[trend]/votes if votes > 0 else 0
            }
            for trend, votes in trend_votes.items()
        }
    }

@router.post("/llm-analysis/predict-all")
async def analyze_market_all_models(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze market data using all available LLM models and provide predictions
    """
    try:
        # Read and process CSV data
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Calculate market metrics
        metrics = calculate_market_metrics(df)
        structure = analyze_market_structure(df)
        
        # Prepare market data description for LLM
        market_description = generate_market_description(df, metrics, structure)
        
        # Get predictions from all models
        all_predictions = {}
        for model_name, model_info in LLM_MODELS.items():
            try:
                prediction = get_llm_prediction(market_description, model_name)
                all_predictions[model_name] = {
                    "name": model_info["name"],
                    "type": model_info["type"],
                    "prediction": prediction
                }
            except Exception as e:
                logging.error(f"Error in model {model_name}: {str(e)}")
                all_predictions[model_name] = {
                    "name": model_info["name"],
                    "type": model_info["type"],
                    "error": str(e)
                }
        
        # Calculate consensus
        consensus = calculate_consensus_prediction(all_predictions)
        
        return {
            "consensus": consensus,
            "model_predictions": all_predictions,
            "metrics": metrics,
            "market_structure": structure,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logging.error(f"Error in market analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during analysis: {str(e)}"
        )

@router.post("/llm-analysis/predict-all")
async def predict_all_models(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze market data using all available LLM models and provide predictions
    """
    try:
        # Read and process CSV data
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Analyze market with all models
        predictions = await analyze_market_all_models(file)
        
        # Calculate consensus prediction
        consensus = calculate_consensus_prediction(predictions)
        
        return {
            "predictions": predictions,
            "consensus": consensus,
            "models": list(LLM_MODELS.keys())
        }
    except Exception as e:
        logging.error(f"Error in predict-all: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
