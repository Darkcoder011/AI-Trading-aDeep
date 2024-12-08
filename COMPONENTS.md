# LLM Market Analysis System - Component Documentation

## Backend Components

### 1. Routes Directory (`/backend/routes/`)

#### 1.1 LLM Analysis (`llm_analysis.py`)
- **Purpose**: Core LLM model management and prediction
- **Key Components**:
  - `get_llm_prediction()`: Generates predictions using specified LLM model
  - `analyze_market()`: Processes market data and returns predictions
  - `analyze_market_all_models()`: Multi-model analysis with consensus
  - `calculate_consensus_prediction()`: Aggregates predictions from multiple models
- **Models**:
  ```python
  LLM_MODELS = {
      "finbert": {
          "name": "finbert-sentiment",
          "type": "sentiment_analysis",
          "description": "Financial sentiment analysis"
      },
      "bert_market": {
          "name": "bert-market",
          "type": "market_prediction",
          "description": "Market pattern recognition"
      }
      # ... other models
  }
  ```
- **Output Format**:
  ```json
  {
      "model_name": "finbert",
      "prediction": {
          "trend": "uptrend",
          "confidence": 0.85,
          "strength": "high",
          "timeframe": "15min"
      }
  }
  ```

#### 1.2 New LLM Models (`llm_new.py`)
- **Purpose**: Enhanced LLM model implementation with technical indicators
- **Key Functions**:
  - `analyze_market_data()`: Single model analysis
  - `analyze_market_data_all_models()`: Multi-model analysis
  - `calculate_technical_indicators()`: RSI, MACD, SMA calculations
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - SMA (Simple Moving Average)
  - Bollinger Bands
- **Response Format**:
  ```json
  {
      "status": "success",
      "predictions": {
          "model_predictions": [...],
          "consensus": {...},
          "technical_indicators": {...}
      }
  }
  ```

#### 1.3 AutoML Analysis (`automl_analysis.py`)
- **Purpose**: Automated machine learning model selection and optimization
- **Models**:
  - TrendMaster XGB: Overall trend analysis
  - VolumeWave LGB: Volume pattern analysis
  - MomentumCat: Momentum indicators
  - PriceForest: Price pattern recognition
  - WaveRider GBM: Wave analysis
  - NeuralMarket: Neural network predictions
  - SupportVector Pro: Support/Resistance levels
  - TrendForce XGB: Trend strength analysis
  - VolumePulse LGB: Volume analysis
  - MarketMind Cat: Market psychology
- **Features**:
  - Automated model selection
  - Hyperparameter optimization
  - Cross-validation
  - Performance metrics

#### 1.4 Prediction Routes (`prediction.py`)
- **Purpose**: Strategy-based market predictions
- **Strategies**:
  1. RSI + MACD
     - Combines momentum and trend
     - Entry/exit signals
  2. Bollinger + Stochastic
     - Volatility and momentum
     - Overbought/oversold conditions
  3. ADX + DMI
     - Trend strength
     - Directional movement
  4. Volume + MFI
     - Volume analysis
     - Money flow patterns
  5. Triple Screen
     - Multiple timeframe analysis
     - Trend confirmation
  6. Other strategies...

## Frontend Components

### 1. Analysis Components

#### 1.1 LLM New Model (`LLMNewModel.jsx`)
- **Purpose**: Main interface for LLM model analysis
- **Features**:
  - File upload with drag-and-drop
  - Model selection interface
  - Results visualization
  - Technical indicator display
- **Components**:
  ```jsx
  - ModelResultsTable
  - TechnicalIndicators
  - MarketDataSummary
  - ConsensusView
  ```
- **State Management**:
  - File handling
  - API integration
  - Results processing
  - Error handling

#### 1.2 LLM Model Analysis (`LLMModelAnalysis.jsx`)
- **Purpose**: Advanced market analysis interface
- **Features**:
  - Multiple timeframe analysis
  - Technical indicator charts
  - Model comparison
  - Prediction history
- **Components**:
  ```jsx
  - TimeframeSelector
  - IndicatorCharts
  - ModelComparison
  - PredictionHistory
  ```

### 2. Utility Components
- Data formatting
- API integration
- Error handling
- Loading states

## Frontend Architecture

### 1. Core Components

#### 1.1 LLM New Model Component
```jsx
// LLMNewModel.jsx
const LLMNewModel = () => {
    // State Management
    const [file, setFile] = useState(null);
    const [models, setModels] = useState([]);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);

    // Component Structure
    return (
        <Box>
            <FileUpload onFileSelect={handleFileUpload} />
            <ModelSelection models={models} onSelect={handleModelSelect} />
            <AnalysisResults results={results} />
            <TechnicalIndicators data={marketData} />
            <ConsensusView consensus={consensusData} />
        </Box>
    );
};

// Sub-components
const FileUpload = styled(Box)({
    border: '2px dashed #ccc',
    padding: '20px',
    textAlign: 'center',
    '&:hover': {
        borderColor: '#2196f3'
    }
});

const ModelSelection = styled(Grid)({
    marginTop: '20px',
    gap: '10px',
    '& .model-card': {
        cursor: 'pointer',
        transition: 'transform 0.2s',
        '&:hover': {
            transform: 'scale(1.02)'
        }
    }
});
```

#### 1.2 Analysis Results Table
```jsx
// Components/Analysis/ModelResultsTable.jsx
const ModelResultsTable = ({ predictions, consensus }) => (
    <TableContainer>
        <Table>
            <TableHead>
                <TableRow>
                    <TableCell>Model</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Trend</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Movement</TableCell>
                </TableRow>
            </TableHead>
            <TableBody>
                {predictions.map((model) => (
                    <ModelResultRow key={model.id} data={model} />
                ))}
                <ConsensusRow data={consensus} />
            </TableBody>
        </Table>
    </TableContainer>
);
```

### 2. Visualization Components

#### 2.1 Technical Indicators Chart
```jsx
// Components/Charts/TechnicalChart.jsx
const TechnicalChart = ({ data }) => {
    const chartConfig = {
        syncId: 'marketAnalysis',
        height: 400,
        margin: { top: 10, right: 30, left: 0, bottom: 0 }
    };

    return (
        <Box>
            <PriceChart data={data} {...chartConfig} />
            <VolumeChart data={data} {...chartConfig} />
            <IndicatorChart 
                data={data}
                indicators={['RSI', 'MACD', 'SMA']}
                {...chartConfig}
            />
        </Box>
    );
};
```

#### 2.2 Market Overview Dashboard
```jsx
// Components/Dashboard/MarketOverview.jsx
const MarketOverview = ({ marketData, predictions }) => (
    <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
            <PriceSummaryCard data={marketData} />
        </Grid>
        <Grid item xs={12} md={6}>
            <TechnicalIndicatorsCard data={marketData} />
        </Grid>
        <Grid item xs={12}>
            <PredictionSummaryCard predictions={predictions} />
        </Grid>
    </Grid>
);
```

### 3. State Management

#### 3.1 Analysis Context
```jsx
// Context/AnalysisContext.jsx
const AnalysisContext = createContext();

export const AnalysisProvider = ({ children }) => {
    const [state, dispatch] = useReducer(analysisReducer, initialState);
    
    const contextValue = {
        marketData: state.marketData,
        predictions: state.predictions,
        indicators: state.indicators,
        loading: state.loading,
        error: state.error,
        dispatch
    };

    return (
        <AnalysisContext.Provider value={contextValue}>
            {children}
        </AnalysisContext.Provider>
    );
};
```

#### 3.2 API Integration
```jsx
// Services/api.js
const api = {
    async analyzeTrade(file, modelId) {
        const formData = new FormData();
        formData.append('file', file);
        
        return axios.post(
            `/api/llm-new/analyze-market/${modelId}`,
            formData,
            {
                headers: { 'Content-Type': 'multipart/form-data' }
            }
        );
    },
    
    async getModels() {
        return axios.get('/api/llm-new/models');
    }
};
```

### 4. Custom Hooks

#### 4.1 Market Analysis Hook
```jsx
// Hooks/useMarketAnalysis.js
const useMarketAnalysis = () => {
    const [analysis, setAnalysis] = useState({
        loading: false,
        data: null,
        error: null
    });

    const analyzeMarket = async (file, modelId) => {
        try {
            setAnalysis(prev => ({ ...prev, loading: true }));
            const result = await api.analyzeTrade(file, modelId);
            setAnalysis({
                loading: false,
                data: result.data,
                error: null
            });
        } catch (error) {
            setAnalysis({
                loading: false,
                data: null,
                error: error.message
            });
        }
    };

    return { analysis, analyzeMarket };
};
```

#### 4.2 Technical Indicators Hook
```jsx
// Hooks/useTechnicalIndicators.js
const useTechnicalIndicators = (marketData) => {
    const [indicators, setIndicators] = useState({
        rsi: [],
        macd: [],
        sma: []
    });

    useEffect(() => {
        if (marketData) {
            setIndicators({
                rsi: calculateRSI(marketData),
                macd: calculateMACD(marketData),
                sma: calculateSMA(marketData)
            });
        }
    }, [marketData]);

    return indicators;
};
```

### 5. Styled Components

#### 5.1 Theme Configuration
```jsx
// Styles/theme.js
const theme = createTheme({
    palette: {
        primary: {
            main: '#1976d2',
            light: '#42a5f5',
            dark: '#1565c0'
        },
        secondary: {
            main: '#dc004e',
            light: '#ff4081',
            dark: '#9a0036'
        }
    },
    components: {
        MuiCard: {
            styleOverrides: {
                root: {
                    borderRadius: 8,
                    boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                }
            }
        }
    }
});
```

#### 5.2 Common Components
```jsx
// Components/Common/index.js
export const LoadingOverlay = styled(Box)({
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.8)',
    zIndex: 1000
});

export const ErrorMessage = styled(Alert)({
    margin: '16px 0',
    borderRadius: 8
});

export const SuccessMessage = styled(Alert)({
    margin: '16px 0',
    borderRadius: 8,
    backgroundColor: '#e8f5e9'
});
```

This frontend architecture provides:
- Modular component structure
- Reusable styled components
- Custom hooks for business logic
- Context-based state management
- Responsive design patterns
- Error handling
- Loading states
- API integration

Would you like me to add more details about any specific component or add additional frontend features?

## API Integration

### 1. Backend Endpoints
```
POST /llm-analysis/predict
POST /llm-analysis/predict-all
POST /llm-new/analyze-market
POST /llm-new/analyze-market-all
POST /automl/analyze
POST /prediction/analyze
```

### 2. Data Flow
1. User uploads market data
2. Data preprocessing
3. Model analysis
4. Results aggregation
5. Frontend display

## Performance Considerations
- Batch processing for large datasets
- Caching for frequent requests
- Optimized model loading
- Efficient data structures

## Security Measures
- Input validation
- Rate limiting
- Error handling
- Data sanitization

## Future Improvements
1. Real-time data streaming
2. Advanced visualization
3. Custom model training
4. Performance optimization
5. Additional technical indicators

## Detailed Model Documentation

### 1. Core LLM Models

#### 1.1 Financial Sentiment Models
1. **FinBERT**
   - **Purpose**: Financial sentiment analysis
   - **Architecture**: BERT-base with financial fine-tuning
   - **Input**: Market data and news
   - **Output**: Sentiment (Bullish/Bearish/Neutral)
   - **Use Case**: Market sentiment prediction

2. **BERT Market**
   - **Purpose**: Market pattern recognition
   - **Architecture**: BERT-large with market data fine-tuning
   - **Input**: Price action patterns
   - **Output**: Pattern classification
   - **Use Case**: Technical pattern identification

3. **GPT-2 Trading**
   - **Purpose**: Trading signal generation
   - **Architecture**: GPT-2 with trading strategy fine-tuning
   - **Input**: Market conditions and indicators
   - **Output**: Trading signals and rationale
   - **Use Case**: Automated trading suggestions

#### 1.2 Market Analysis Models
4. **RoBERTa Market**
   - **Purpose**: Advanced market analysis
   - **Architecture**: RoBERTa with market optimization
   - **Input**: Multi-timeframe market data
   - **Output**: Comprehensive market analysis
   - **Use Case**: Complex pattern recognition

5. **XLNet Market**
   - **Purpose**: Sequential market prediction
   - **Architecture**: XLNet with time-series adaptation
   - **Input**: Historical price sequences
   - **Output**: Future price movement predictions
   - **Use Case**: Time-series forecasting

6. **DistilBERT Market**
   - **Purpose**: Lightweight market analysis
   - **Architecture**: Distilled BERT for efficiency
   - **Input**: Basic market indicators
   - **Output**: Quick market assessments
   - **Use Case**: Real-time analysis

#### 1.3 Specialized Models
7. **ALBERT Market**
   - **Purpose**: Efficient market modeling
   - **Architecture**: ALBERT with parameter sharing
   - **Input**: Market microstructure data
   - **Output**: Market efficiency metrics
   - **Use Case**: Market microstructure analysis

8. **ELECTRA Market**
   - **Purpose**: Market pattern detection
   - **Architecture**: ELECTRA with discriminative training
   - **Input**: Price and volume patterns
   - **Output**: Pattern authenticity scores
   - **Use Case**: False pattern detection

9. **Longformer Market**
   - **Purpose**: Long-term trend analysis
   - **Architecture**: Longformer with extended context
   - **Input**: Extended historical data
   - **Output**: Long-term trend predictions
   - **Use Case**: Long-term investment analysis

10. **DeBERTa Market**
    - **Purpose**: Advanced market sentiment
    - **Architecture**: DeBERTa with enhanced attention
    - **Input**: Market sentiment indicators
    - **Output**: Refined sentiment analysis
    - **Use Case**: High-precision sentiment tracking

### 2. Trading Strategies

#### 2.1 Technical Analysis Strategies
1. **RSI + MACD Strategy**
   ```python
   {
       "name": "RSI_MACD_Strategy",
       "components": {
           "RSI": {"period": 14, "overbought": 70, "oversold": 30},
           "MACD": {"fast": 12, "slow": 26, "signal": 9}
       },
       "rules": {
           "buy": "RSI < 30 and MACD > Signal",
           "sell": "RSI > 70 and MACD < Signal"
       }
   }
   ```

2. **Bollinger + Stochastic Strategy**
   ```python
   {
       "name": "Bollinger_Stoch_Strategy",
       "components": {
           "Bollinger": {"period": 20, "std_dev": 2},
           "Stochastic": {"k_period": 14, "d_period": 3}
       },
       "rules": {
           "buy": "Price < Lower_Band and Stoch_K < 20",
           "sell": "Price > Upper_Band and Stoch_K > 80"
       }
   }
   ```

#### 2.2 Volume Analysis Strategies
3. **Volume + MFI Strategy**
   ```python
   {
       "name": "Volume_MFI_Strategy",
       "components": {
           "Volume": {"sma_period": 20},
           "MFI": {"period": 14, "threshold": 20}
       },
       "rules": {
           "buy": "Volume > Vol_SMA * 1.5 and MFI < 20",
           "sell": "Volume > Vol_SMA * 1.5 and MFI > 80"
       }
   }
   ```

4. **Triple Screen Strategy**
   ```python
   {
       "name": "Triple_Screen_Strategy",
       "components": {
           "Weekly": {"EMA": 13},
           "Daily": {"MACD": {"fast": 12, "slow": 26}},
           "Hourly": {"Force_Index": 13}
       },
       "rules": {
           "buy": [
               "Weekly_Trend == Bullish",
               "Daily_MACD == Positive",
               "Hourly_Force_Index < 0"
           ]
       }
   }
   ```

#### 2.3 Advanced Strategies
5. **Support Resistance Strategy**
   ```python
   {
       "name": "Support_Resistance_Strategy",
       "components": {
           "Pivot_Points": {"method": "fibonacci"},
           "Volume_Profile": {"period": "session"},
           "Price_Action": {"candle_patterns": True}
       },
       "rules": {
           "support_levels": "Calculate_Dynamic_Support()",
           "resistance_levels": "Calculate_Dynamic_Resistance()",
           "validation": "Volume_Confirmation()"
       }
   }
   ```

6. **Volatility Breakout Strategy**
   ```python
   {
       "name": "Volatility_Breakout",
       "components": {
           "ATR": {"period": 14},
           "Bollinger": {"period": 20, "std_dev": 2},
           "Volume": {"threshold": 1.5}
       },
       "rules": {
           "breakout_up": "Price > Upper_Band and Volume > Vol_Threshold",
           "breakout_down": "Price < Lower_Band and Volume > Vol_Threshold"
       }
   }
   ```

### 3. Model Integration

#### 3.1 Consensus Mechanism
```python
def calculate_consensus(predictions: List[Dict]):
    weights = {
        "finbert": 0.15,
        "bert_market": 0.15,
        "gpt2_trading": 0.12,
        "roberta_market": 0.12,
        "xlnet_market": 0.10,
        "distilbert_market": 0.08,
        "albert_market": 0.08,
        "electra_market": 0.08,
        "longformer_market": 0.07,
        "deberta_market": 0.05
    }
    
    consensus = {
        "trend": weighted_vote(predictions, weights),
        "confidence": weighted_average(predictions, weights),
        "timeframe": "15min"
    }
    return consensus
```

#### 3.2 Strategy Combination
```python
def combine_strategies(strategies: List[Dict]):
    return {
        "entry_signals": combine_entry_signals(strategies),
        "exit_signals": combine_exit_signals(strategies),
        "position_sizing": calculate_position_size(strategies),
        "risk_management": apply_risk_rules(strategies)
    }
