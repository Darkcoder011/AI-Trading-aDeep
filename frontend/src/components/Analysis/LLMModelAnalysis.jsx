import React, { useState } from 'react';
import axios from 'axios';
import {
    Card,
    CardContent,
    Typography,
    Button,
    CircularProgress,
    Grid,
    Box,
    Alert,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    Fade,
    Grow,
    Zoom,
    LinearProgress,
    Divider
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import PsychologyIcon from '@mui/icons-material/Psychology';

const StyledCard = styled(Card)(({ theme }) => ({
    background: 'linear-gradient(135deg, #1a237e 0%, #0d47a1 100%)',
    color: 'white',
    borderRadius: '16px',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
    transition: 'transform 0.3s ease-in-out',
    '&:hover': {
        transform: 'translateY(-5px)'
    }
}));

const AnimatedButton = styled(Button)(({ theme }) => ({
    background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
    border: 0,
    borderRadius: 3,
    boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
    color: 'white',
    height: 48,
    padding: '0 30px',
    transition: 'all 0.3s ease-in-out',
    '&:hover': {
        transform: 'scale(1.05)',
        boxShadow: '0 6px 10px 4px rgba(33, 203, 243, .3)',
    }
}));

const getTrendIcon = (trend) => {
    switch (trend?.toLowerCase()) {
        case 'uptrend':
            return <TrendingUpIcon sx={{ color: '#4caf50' }} />;
        case 'downtrend':
            return <TrendingDownIcon sx={{ color: '#f44336' }} />;
        default:
            return <TrendingFlatIcon sx={{ color: '#ff9800' }} />;
    }
};

const getAgreementLevelColor = (level) => {
    switch (level) {
        case 'strong':
            return '#4caf50';
        case 'moderate':
            return '#ff9800';
        case 'weak':
            return '#f44336';
        default:
            return '#757575';
    }
};

const getTrendEmoji = (trend) => {
    switch (trend?.toLowerCase()) {
        case 'strong_uptrend':
            return '🚀';
        case 'weak_uptrend':
            return '📈';
        case 'strong_downtrend':
            return '📉';
        case 'weak_downtrend':
            return '⬇️';
        case 'sideways':
            return '↔️';
        default:
            return '❓';
    }
};

const getModelTypeEmoji = (type) => {
    switch (type) {
        case 'sentiment':
            return '🎭';
        case 'market_analysis':
            return '📊';
        case 'prediction':
            return '🔮';
        case 'technical_analysis':
            return '📐';
        case 'volume_analysis':
            return '📊';
        case 'momentum_analysis':
            return '🌊';
        case 'support_resistance':
            return '🎯';
        case 'trend_analysis':
            return '📈';
        case 'long_term_patterns':
            return '🔍';
        case 'market_sentiment':
            return '🧠';
        default:
            return '📱';
    }
};

const ModelPredictionTable = ({ predictions }) => {
    if (!predictions || Object.keys(predictions).length === 0) {
        return null;
    }

    return (
        <TableContainer component={Paper} sx={{ mt: 3 }}>
            <Table>
                <TableHead>
                    <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>15min Prediction</TableCell>
                        <TableCell>Confidence</TableCell>
                        <TableCell>Probabilities</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {Object.entries(predictions).map(([modelName, data]) => {
                        const prediction = data.prediction || {};
                        const probabilities = prediction.direction_probabilities || {};
                        
                        return (
                            <TableRow key={modelName} hover>
                                <TableCell>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        {getModelTypeEmoji(data.type)}
                                        <Typography variant="body2">
                                            {data.name}
                                            <Typography variant="caption" display="block" color="textSecondary">
                                                {data.description}
                                            </Typography>
                                        </Typography>
                                    </Box>
                                </TableCell>
                                <TableCell>{data.type}</TableCell>
                                <TableCell>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        {getTrendEmoji(prediction.trend)}
                                        {prediction.trend || 'N/A'}
                                        <Chip 
                                            size="small"
                                            label={prediction.strength || 'N/A'}
                                            color={
                                                prediction.strength === 'high' ? 'success' :
                                                prediction.strength === 'medium' ? 'warning' : 'error'
                                            }
                                        />
                                    </Box>
                                </TableCell>
                                <TableCell>
                                    {prediction.confidence ? (
                                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                            <LinearProgress
                                                variant="determinate"
                                                value={prediction.confidence * 100}
                                                sx={{ 
                                                    width: 100,
                                                    mr: 1,
                                                    height: 8,
                                                    borderRadius: 4
                                                }}
                                            />
                                            {(prediction.confidence * 100).toFixed(1)}%
                                        </Box>
                                    ) : 'N/A'}
                                </TableCell>
                                <TableCell>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                        {Object.entries(probabilities).map(([direction, prob]) => (
                                            <Typography key={direction} variant="caption" sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <span>{direction}:</span>
                                                <span>{(prob * 100).toFixed(1)}%</span>
                                            </Typography>
                                        ))}
                                    </Box>
                                </TableCell>
                            </TableRow>
                        );
                    })}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

const ConsensusCard = ({ consensus }) => {
    if (!consensus) return null;

    return (
        <Grow in={true} timeout={1000}>
            <Paper sx={{ p: 3, mt: 3, background: 'linear-gradient(135deg, #1a237e 0%, #0d47a1 100%)', color: 'white' }}>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    🎯 15-Minute Market Consensus
                    <Chip
                        size="small"
                        label={consensus.agreement_level}
                        color={
                            consensus.agreement_level === 'strong' ? 'success' :
                            consensus.agreement_level === 'moderate' ? 'warning' : 'error'
                        }
                        sx={{ ml: 2 }}
                    />
                </Typography>
                
                <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={12} md={4}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {getTrendEmoji(consensus.trend)}
                            <Typography>
                                Predicted Trend: {consensus.trend}
                            </Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <Typography>
                            Confidence: {(consensus.confidence * 100).toFixed(1)}%
                        </Typography>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <Typography>
                            Timestamp: {new Date(consensus.timestamp).toLocaleTimeString()}
                        </Typography>
                    </Grid>
                </Grid>

                <Divider sx={{ my: 2, bgcolor: 'rgba(255,255,255,0.1)' }} />
                
                <Typography variant="subtitle2" gutterBottom>
                    Model Distribution
                </Typography>
                <Grid container spacing={1}>
                    {Object.entries(consensus.model_distribution || {}).map(([trend, data]) => (
                        <Grid item xs={12} sm={6} md={4} key={trend}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="body2">
                                    {getTrendEmoji(trend)} {trend}
                                </Typography>
                                <Typography variant="body2">
                                    {data.votes} votes ({(data.percentage * 100).toFixed(1)}%)
                                </Typography>
                            </Box>
                            <LinearProgress
                                variant="determinate"
                                value={data.percentage * 100}
                                sx={{ 
                                    height: 4,
                                    borderRadius: 2,
                                    bgcolor: 'rgba(255,255,255,0.1)',
                                    '& .MuiLinearProgress-bar': {
                                        bgcolor: 'rgba(255,255,255,0.8)'
                                    }
                                }}
                            />
                        </Grid>
                    ))}
                </Grid>
            </Paper>
        </Grow>
    );
};

const PressureAnalysisCard = ({ metrics }) => {
    if (!metrics || !metrics.pressure_analysis) {
        return null;  // Don't render if no metrics data
    }

    const pressureData = metrics.pressure_analysis || {};
    
    const chartData = [
        {
            name: 'Buying Pressure',
            value: pressureData.buying_pressure || 0,
            color: '#4caf50'
        },
        {
            name: 'Selling Pressure',
            value: pressureData.selling_pressure || 0,
            color: '#f44336'
        }
    ];

    return (
        <Zoom in={true} timeout={1000}>
            <Paper sx={{ p: 3, mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                    📊 Market Pressure Analysis
                </Typography>
                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <ResponsiveContainer width="100%" height={200}>
                            <BarChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="value" fill="#8884d8" />
                            </BarChart>
                        </ResponsiveContainer>
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <Box>
                            <Typography variant="subtitle1">
                                Pressure Ratio: {(pressureData.pressure_ratio || 0).toFixed(2)}
                            </Typography>
                            <Typography variant="subtitle1">
                                Net Money Flow: {(pressureData.net_money_flow || 0).toFixed(2)}
                            </Typography>
                            <Typography variant="subtitle1">
                                RSI: {(pressureData.rsi || 0).toFixed(2)}
                            </Typography>
                        </Box>
                    </Grid>
                </Grid>
            </Paper>
        </Zoom>
    );
};

const LLMModelAnalysis = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file && file.type === "text/csv") {
            setSelectedFile(file);
            setError(null);
        } else {
            setError("Please upload a valid CSV file");
        }
    };

    const handleAnalysis = async () => {
        if (!selectedFile) {
            setError("Please select a CSV file first");
            return;
        }

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await axios.post(
                'http://localhost:8000/api/v1/llm-analysis/predict',
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                }
            );
            
            console.log('Full API Response:', response.data);
            
            if (!response.data || typeof response.data !== 'object') {
                setError('Received invalid response from server');
                return;
            }

            const analysisResult = {
                model_name: response.data.model_name || 'Unknown Model',
                trend: response.data.trend || 'No trend detected',
                confidence: response.data.confidence || 0,
                timeframe: response.data.timeframe || 'Unknown',
                
                // Ensure volume_analysis has default structure
                volume_analysis: {
                    average_volume: response.data.volume_analysis?.average_volume || 0,
                    volume_trend: response.data.volume_analysis?.volume_trend || 'Neutral',
                    volume_strength: response.data.volume_analysis?.volume_strength || 0,
                    volume_volatility: response.data.volume_analysis?.volume_volatility || 0
                },
                
                // Ensure pressure_analysis has default structure
                pressure_analysis: {
                    buying_pressure: response.data.pressure_analysis?.buying_pressure || 0,
                    selling_pressure: response.data.pressure_analysis?.selling_pressure || 0,
                    pressure_ratio: response.data.pressure_analysis?.pressure_ratio || 0,
                    buying_volume_avg: response.data.pressure_analysis?.buying_volume_avg || 0,
                    selling_volume_avg: response.data.pressure_analysis?.selling_volume_avg || 0,
                    net_money_flow: response.data.pressure_analysis?.net_money_flow || 0,
                    pressure_momentum: response.data.pressure_analysis?.pressure_momentum || 0,
                    rsi: response.data.pressure_analysis?.rsi || 0
                },
                
                // Ensure support_resistance has default structure
                support_resistance: {
                    support_levels: response.data.support_resistance?.support_levels || [],
                    resistance_levels: response.data.support_resistance?.resistance_levels || []
                },
                
                // Ensure market_structure has default structure
                market_structure: {
                    market_phase: response.data.market_structure?.market_phase || 'Undefined',
                    support_levels: response.data.market_structure?.support_levels || [],
                    resistance_levels: response.data.market_structure?.resistance_levels || [],
                    pattern_probability: response.data.market_structure?.pattern_probability || {}
                },
                
                // Ensure predictions has default structure for model prediction tables
                predictions: response.data.predictions || {}
            };

            console.log('Processed Analysis Result:', analysisResult);

            setAnalysisResult(analysisResult);
        } catch (err) {
            console.error('Full Error Object:', err);
            setError(
                err.response?.data?.detail || 
                err.message || 
                'Unexpected error during analysis'
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
        >
            <StyledCard sx={{ margin: '20px' }}>
                <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                        <PsychologyIcon sx={{ fontSize: 40 }} />
                        <Typography variant="h4" component="div">
                            🤖 Multi-Model Market Analysis
                        </Typography>
                    </Box>

                    <Fade in={true} timeout={1000}>
                        <Box>
                            <AnimatedButton
                                variant="contained"
                                component="label"
                                fullWidth
                                startIcon={<CloudUploadIcon />}
                            >
                                📊 Upload Market Data (CSV)
                                <input
                                    type="file"
                                    hidden
                                    accept=".csv"
                                    onChange={handleFileUpload}
                                />
                            </AnimatedButton>

                            {selectedFile && (
                                <Typography variant="body2" sx={{ mt: 2, color: '#90caf9' }}>
                                    📁 Selected file: {selectedFile.name}
                                </Typography>
                            )}

                            <AnimatedButton
                                variant="contained"
                                onClick={handleAnalysis}
                                disabled={loading || !selectedFile}
                                fullWidth
                                sx={{ mt: 2 }}
                                startIcon={<AutoGraphIcon />}
                            >
                                {loading ? (
                                    <CircularProgress size={24} color="inherit" />
                                ) : (
                                    '🔮 Analyze with All Models'
                                )}
                            </AnimatedButton>
                        </Box>
                    </Fade>

                    {error && (
                        <Zoom in={true}>
                            <Alert severity="error" sx={{ mt: 2 }}>
                                ❌ {error}
                            </Alert>
                        </Zoom>
                    )}

                    {analysisResult && (
                        <Fade in={true} timeout={1000}>
                            <Box sx={{ mt: 3 }}>
                                <ConsensusCard consensus={analysisResult.consensus} />
                                
                                <Zoom in={true} timeout={1000}>
                                    <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>
                                        📊 Individual Model Predictions
                                    </Typography>
                                </Zoom>
                                
                                <ModelPredictionTable predictions={analysisResult.predictions} />

                                <Grow in={true} timeout={1500}>
                                    <Paper sx={{ p: 3, mt: 3 }}>
                                        <Typography variant="h6" gutterBottom>
                                            📈 Market Structure
                                        </Typography>
                                        <Typography>
                                            Current Phase: {analysisResult.market_structure.market_phase}
                                        </Typography>
                                        <Box sx={{ mt: 2 }}>
                                            <Typography variant="subtitle1" gutterBottom>
                                                Support Levels:
                                            </Typography>
                                            {analysisResult.market_structure.support_levels.map((level, index) => (
                                                <Chip
                                                    key={index}
                                                    label={`S${index + 1}: ${level.toFixed(2)}`}
                                                    sx={{ m: 0.5 }}
                                                    color="primary"
                                                />
                                            ))}
                                        </Box>
                                        <Box sx={{ mt: 2 }}>
                                            <Typography variant="subtitle1" gutterBottom>
                                                Resistance Levels:
                                            </Typography>
                                            {analysisResult.market_structure.resistance_levels.map((level, index) => (
                                                <Chip
                                                    key={index}
                                                    label={`R${index + 1}: ${level.toFixed(2)}`}
                                                    sx={{ m: 0.5 }}
                                                    color="secondary"
                                                />
                                            ))}
                                        </Box>
                                    </Paper>
                                </Grow>

                                <PressureAnalysisCard metrics={analysisResult} />
                            </Box>
                        </Fade>
                    )}
                </CardContent>
            </StyledCard>
        </motion.div>
    );
};

export default LLMModelAnalysis;
