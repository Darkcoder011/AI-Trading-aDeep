from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import pandas as pd

router = APIRouter()

class NewModelRequest(BaseModel):
    name: str
    type: str
    description: str
    endpoint: str
    parameters: Optional[Dict[str, Any]] = {}

# Store for new models (in production, this should be a database)
NEW_MODELS = {}

# Pre-defined LLM models
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

# Pre-defined open source LLM models
OPEN_SOURCE_MODELS = {
    "gpt2-financial": {
        "name": "GPT-2 Financial",
        "type": "transformer",
        "description": "Fine-tuned GPT-2 model for financial analysis",
        "endpoint": "huggingface/gpt2-financial",
        "parameters": {
            "max_length": 100,
            "temperature": 0.7
        }
    },
    "bert-finance": {
        "name": "BERT Finance",
        "type": "transformer",
        "description": "BERT model specialized for financial market analysis",
        "endpoint": "huggingface/bert-finance",
        "parameters": {
            "max_length": 512
        }
    },
    "finbert": {
        "name": "FinBERT",
        "type": "transformer",
        "description": "BERT model fine-tuned on financial text",
        "endpoint": "huggingface/finbert",
        "parameters": {
            "max_length": 512
        }
    }
}

@router.post("/llm-new/add-model", tags=["llm"])
async def add_new_model(model_data: NewModelRequest) -> Dict[str, Any]:
    """
    Add a new LLM model to the system
    """
    try:
        # Validate model name uniqueness
        if model_data.name in NEW_MODELS:
            return {
                "status": "error",
                "detail": f"Model with name {model_data.name} already exists"
            }
        
        # Add timestamp and prepare model data
        model_entry = {
            **model_data.dict(),
            "added_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Store the new model
        NEW_MODELS[model_data.name] = model_entry
        
        logging.info(f"Successfully added new model: {model_data.name}")
        return {
            "status": "success",
            "message": f"Model {model_data.name} added successfully",
            "model": model_entry
        }
        
    except Exception as e:
        logging.error(f"Error adding new model: {str(e)}")
        return {
            "status": "error",
            "detail": f"Error adding new model: {str(e)}"
        }

@router.get("/llm-new/models")
async def get_new_models() -> Dict[str, Any]:
    """
    Get list of all added models
    """
    try:
        return {
            "status": "success",
            "models": list(NEW_MODELS.values())
        }
    except Exception as e:
        logging.error(f"Error retrieving models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving models: {str(e)}"
        )

@router.get("/llm-new/open-source-models")
async def get_open_source_models() -> Dict[str, Any]:
    """
    Get list of available open source models
    """
    try:
        return {
            "status": "success",
            "models": OPEN_SOURCE_MODELS
        }
    except Exception as e:
        logging.error(f"Error retrieving open source models: {str(e)}")
        return {
            "status": "error",
            "detail": f"Error retrieving models: {str(e)}"
        }

@router.post("/llm-new/analyze-market")
async def analyze_market_data(
    file: UploadFile,
    model_name: str,
) -> Dict[str, Any]:
    """
    Analyze market data using selected LLM model and provide predictions
    """
    try:
        if model_name not in OPEN_SOURCE_MODELS:
            return {
                "status": "error",
                "detail": "Invalid model selection"
            }

        # Read CSV data
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Ensure we have the last 100 candles
        df = df.tail(100)
        
        # Calculate basic technical indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'] = calculate_macd(df['close'])
        
        # Prepare market description
        market_description = f"""
        Analyze the following market data for the last 100 candles:
        - Current Price: {df['close'].iloc[-1]:.2f}
        - Price Change: {((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100):.2f}%
        - 20 SMA: {df['SMA_20'].iloc[-1]:.2f}
        - RSI: {df['RSI'].iloc[-1]:.2f}
        - MACD: {df['MACD'].iloc[-1]:.2f}
        
        Based on this data, predict the market movement for the next 10 minutes:
        1. Trend (Uptrend/Downtrend/Sideways)
        2. Confidence level (0-100%)
        3. Price movement percentage expectation
        """
        
        # Get model prediction
        prediction = get_model_prediction(market_description, model_name)
        
        return {
            "status": "success",
            "model_name": model_name,
            "prediction": {
                "trend": prediction["trend"],
                "confidence": prediction["confidence"],
                "price_movement": prediction["price_movement"],
                "analysis_time": datetime.now().isoformat()
            },
            "market_data": {
                "current_price": df['close'].iloc[-1],
                "price_change": ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100),
                "technical_indicators": {
                    "sma_20": df['SMA_20'].iloc[-1],
                    "rsi": df['RSI'].iloc[-1],
                    "macd": df['MACD'].iloc[-1]
                }
            }
        }
        
    except Exception as e:
        logging.error(f"Error in market analysis: {str(e)}")
        return {
            "status": "error",
            "detail": f"Error analyzing market data: {str(e)}"
        }

@router.post("/llm-new/analyze-market-all")
async def analyze_market_data_all_models(
    file: UploadFile
) -> Dict[str, Any]:
    """
    Analyze market data using all available LLM models
    """
    try:
        # Read CSV data
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Ensure we have the last 100 candles
        df = df.tail(100)
        
        # Calculate technical indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['RSI'] = calculate_rsi(df['close'])
        df['MACD'] = calculate_macd(df['close'])
        
        # Prepare market description
        market_description = f"""
        Analyze the following market data for the last 100 candles:
        - Current Price: {df['close'].iloc[-1]:.2f}
        - Price Change: {((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100):.2f}%
        - 20 SMA: {df['SMA_20'].iloc[-1]:.2f}
        - RSI: {df['RSI'].iloc[-1]:.2f}
        - MACD: {df['MACD'].iloc[-1]:.2f}
        
        Based on this data, predict the market movement for the next 10 minutes:
        1. Trend (Uptrend/Downtrend/Sideways)
        2. Confidence level (0-100%)
        3. Price movement percentage expectation
        """
        
        # Get predictions from all models
        all_predictions = {}
        for model_id, model_info in LLM_MODELS.items():
            prediction = get_model_prediction(market_description, model_id)
            all_predictions[model_id] = {
                **model_info,
                "prediction": prediction
            }
        
        # Calculate consensus
        trends = [pred["prediction"]["trend"] for pred in all_predictions.values()]
        confidences = [pred["prediction"]["confidence"] for pred in all_predictions.values()]
        movements = [pred["prediction"]["price_movement"] for pred in all_predictions.values()]
        
        consensus = {
            "trend": max(set(trends), key=trends.count),
            "confidence": sum(confidences) / len(confidences),
            "price_movement": sum(movements) / len(movements)
        }
        
        return {
            "status": "success",
            "predictions": all_predictions,
            "consensus": consensus,
            "market_data": {
                "current_price": df['close'].iloc[-1],
                "price_change": ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100),
                "technical_indicators": {
                    "sma_20": df['SMA_20'].iloc[-1],
                    "rsi": df['RSI'].iloc[-1],
                    "macd": df['MACD'].iloc[-1]
                }
            }
        }
        
    except Exception as e:
        logging.error(f"Error in market analysis: {str(e)}")
        return {
            "status": "error",
            "detail": f"Error analyzing market data: {str(e)}"
        }

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD technical indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def get_model_prediction(market_description: str, model_name: str) -> Dict[str, Any]:
    """Get prediction from the selected model"""
    # Simulate model prediction (replace with actual model inference)
    import random
    
    trends = ["Uptrend", "Downtrend", "Sideways"]
    trend = random.choice(trends)
    confidence = random.uniform(0.6, 0.95)
    price_movement = random.uniform(-2.0, 2.0)
    
    return {
        "trend": trend,
        "confidence": confidence * 100,  # Convert to percentage
        "price_movement": price_movement
    }
