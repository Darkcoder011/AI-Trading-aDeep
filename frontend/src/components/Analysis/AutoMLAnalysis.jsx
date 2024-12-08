import React, { useState } from 'react';
import {
    Box,
    Paper,
    Typography,
    Button,
    Grid,
    CircularProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Fade,
    Grow,
    LinearProgress,
    Alert,
    Chip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';

const VisuallyHiddenInput = styled('input')`
    clip: rect(0 0 0 0);
    clip-path: inset(50%);
    height: 1px;
    overflow: hidden;
    position: absolute;
    white-space: nowrap;
    width: 1px;
`;

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

const ModelCard = ({ model, metrics }) => (
    <Grow in={true} timeout={1000}>
        <Paper sx={{ 
            p: 3, 
            background: 'linear-gradient(135deg, #1a237e 0%, #283593 100%)', 
            color: 'white',
            mb: 2
        }}>
            <Typography variant="h6" gutterBottom>
                {model === 'xgboost' ? '🚀 XGBoost' : 
                 model === 'lightgbm' ? '⚡ LightGBM' : 
                 model === 'catboost' ? '😺 CatBoost' : 
                 '🤖 AutoML'} Performance
            </Typography>
            <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                    <Box>
                        <Typography variant="subtitle2">Accuracy</Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <LinearProgress
                                variant="determinate"
                                value={metrics.accuracy * 100}
                                sx={{
                                    width: '100%',
                                    height: 10,
                                    borderRadius: 5,
                                    backgroundColor: 'rgba(255,255,255,0.2)',
                                    '& .MuiLinearProgress-bar': {
                                        borderRadius: 5,
                                        background: 'linear-gradient(45deg, #00e676 30%, #69f0ae 90%)',
                                    }
                                }}
                            />
                            <Typography>{(metrics.accuracy * 100).toFixed(2)}%</Typography>
                        </Box>
                    </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="subtitle2">F1 Score</Typography>
                    <Typography>{metrics.f1_score.toFixed(3)} 📊</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="subtitle2">Precision</Typography>
                    <Typography>{metrics.precision.toFixed(3)} 🎯</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="subtitle2">Recall</Typography>
                    <Typography>{metrics.recall.toFixed(3)} 🔍</Typography>
                </Grid>
            </Grid>
        </Paper>
    </Grow>
);

const ConsensusPredictionCard = ({ prediction }) => {
    const getTrendColor = (trend) => {
        switch (trend) {
            case 'strong_uptrend':
                return '#00e676';
            case 'weak_uptrend':
                return '#69f0ae';
            case 'strong_downtrend':
                return '#ff1744';
            case 'weak_downtrend':
                return '#ff5252';
            default:
                return '#ffeb3b';
        }
    };

    return (
        <Grow in={true} timeout={1000}>
            <Paper sx={{ 
                p: 3, 
                background: `linear-gradient(135deg, ${getTrendColor(prediction.trend)} 0%, ${getTrendColor(prediction.trend)}80 100%)`,
                color: 'white',
                mb: 3
            }}>
                <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getTrendEmoji(prediction.trend)} Consensus Prediction
                </Typography>
                <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                        <Typography variant="subtitle1">Next 10min Trend</Typography>
                        <Typography variant="h6">
                            {prediction.trend.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                        </Typography>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <Typography variant="subtitle1">Confidence Level</Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <CircularProgress
                                variant="determinate"
                                value={prediction.confidence_level * 100}
                                sx={{
                                    color: 'white',
                                    '& .MuiCircularProgress-circle': {
                                        strokeLinecap: 'round',
                                    },
                                }}
                            />
                            <Typography variant="h6">
                                {(prediction.confidence_level * 100).toFixed(1)}%
                            </Typography>
                        </Box>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <Typography variant="subtitle1">Prediction Time</Typography>
                        <Typography variant="h6">
                            {new Date(prediction.next_timestamp).toLocaleTimeString()} 🕙
                        </Typography>
                    </Grid>
                </Grid>
            </Paper>
        </Grow>
    );
};

const PressureAnalysisCard = ({ pressure }) => (
    <Grow in={true} timeout={1000}>
        <Paper sx={{ 
            p: 3, 
            background: 'linear-gradient(135deg, #311b92 0%, #4527a0 100%)',
            color: 'white',
            mb: 3
        }}>
            <Typography variant="h6" gutterBottom>
                💪 Market Pressure Analysis
            </Typography>
            <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1">
                        Buying Pressure 🛍️
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                            variant="determinate"
                            value={pressure.buying_pressure * 100}
                            sx={{ 
                                width: '100%',
                                height: 20,
                                borderRadius: 10,
                                backgroundColor: 'rgba(255,255,255,0.2)',
                                '& .MuiLinearProgress-bar': {
                                    borderRadius: 10,
                                    background: 'linear-gradient(45deg, #00e676 30%, #69f0ae 90%)',
                                }
                            }}
                        />
                        <Typography>
                            {(pressure.buying_pressure * 100).toFixed(1)}%
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1">
                        Selling Pressure 📉
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                            variant="determinate"
                            value={pressure.selling_pressure * 100}
                            sx={{ 
                                width: '100%',
                                height: 20,
                                borderRadius: 10,
                                backgroundColor: 'rgba(255,255,255,0.2)',
                                '& .MuiLinearProgress-bar': {
                                    borderRadius: 10,
                                    background: 'linear-gradient(45deg, #ff1744 30%, #ff5252 90%)',
                                }
                            }}
                        />
                        <Typography>
                            {(pressure.selling_pressure * 100).toFixed(1)}%
                        </Typography>
                    </Box>
                </Grid>
                <Grid item xs={12}>
                    <Box sx={{ 
                        mt: 2, 
                        p: 2, 
                        borderRadius: 2,
                        backgroundColor: 'rgba(255,255,255,0.1)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1
                    }}>
                        <Typography variant="subtitle1">
                            Net Pressure:
                        </Typography>
                        <Typography variant="h6" sx={{ 
                            color: pressure.net_pressure > 0 ? '#00e676' : '#ff1744'
                        }}>
                            {pressure.net_pressure > 0 ? '📈' : '📉'} {(pressure.net_pressure * 100).toFixed(1)}%
                        </Typography>
                    </Box>
                </Grid>
            </Grid>
        </Paper>
    </Grow>
);

const ModelGrid = ({ models }) => {
    // Sort models by accuracy
    const sortedModels = [...models].sort((a, b) => b.accuracy - a.accuracy);
    
    return (
        <Grid container spacing={2}>
            {sortedModels.map((model, index) => (
                <Grid item xs={12} md={6} key={index}>
                    <Grow in={true} timeout={1000 + index * 100}>
                        <Paper sx={{ 
                            p: 2,
                            background: 'linear-gradient(135deg, #1a237e 0%, #283593 100%)',
                            color: 'white',
                            height: '100%'
                        }}>
                            <Typography variant="h6" gutterBottom sx={{ 
                                display: 'flex', 
                                alignItems: 'center', 
                                gap: 1,
                                color: model.accuracy > 0.7 ? '#69f0ae' : 
                                       model.accuracy > 0.6 ? '#ffeb3b' : 'white'
                            }}>
                                {model.emoji} {model.name}
                            </Typography>
                            <Typography variant="subtitle2" gutterBottom sx={{ opacity: 0.8 }}>
                                {model.specialization}
                            </Typography>
                            <Grid container spacing={2}>
                                <Grid item xs={12}>
                                    <Typography variant="body2">Accuracy</Typography>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <LinearProgress
                                            variant="determinate"
                                            value={model.accuracy * 100}
                                            sx={{ 
                                                width: '100%',
                                                height: 8,
                                                borderRadius: 4,
                                                backgroundColor: 'rgba(255,255,255,0.2)',
                                                '& .MuiLinearProgress-bar': {
                                                    borderRadius: 4,
                                                    background: 'linear-gradient(45deg, #00e676 30%, #69f0ae 90%)',
                                                }
                                            }}
                                        />
                                        <Typography variant="body2">
                                            {(model.accuracy * 100).toFixed(1)}%
                                        </Typography>
                                    </Box>
                                </Grid>
                                <Grid item xs={12}>
                                    <Box sx={{ 
                                        display: 'flex', 
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        mt: 1
                                    }}>
                                        <Typography variant="body2">Next Prediction:</Typography>
                                        <Box sx={{ 
                                            display: 'flex', 
                                            alignItems: 'center', 
                                            gap: 1,
                                            backgroundColor: 'rgba(255,255,255,0.1)',
                                            padding: '4px 8px',
                                            borderRadius: '4px'
                                        }}>
                                            {getTrendEmoji(model.next_prediction)}
                                            <Typography variant="body2" sx={{ 
                                                color: model.confidence > 0.7 ? '#69f0ae' : 
                                                       model.confidence > 0.5 ? '#ffeb3b' : '#ff5252'
                                            }}>
                                                {(model.confidence * 100).toFixed(1)}%
                                            </Typography>
                                        </Box>
                                    </Box>
                                </Grid>
                                <Grid item xs={12}>
                                    <Box sx={{ 
                                        display: 'flex', 
                                        justifyContent: 'space-between',
                                        mt: 1
                                    }}>
                                        <Typography variant="caption" sx={{ opacity: 0.7 }}>
                                            F1: {model.f1_score.toFixed(3)}
                                        </Typography>
                                        <Typography variant="caption" sx={{ opacity: 0.7 }}>
                                            Precision: {model.precision.toFixed(3)}
                                        </Typography>
                                        <Typography variant="caption" sx={{ opacity: 0.7 }}>
                                            Recall: {model.recall.toFixed(3)}
                                        </Typography>
                                    </Box>
                                </Grid>
                            </Grid>
                        </Paper>
                    </Grow>
                </Grid>
            ))}
        </Grid>
    );
};

const CombinedPredictionsTable = ({ combinedPredictions }) => {
    if (!combinedPredictions || !combinedPredictions.trends) return null;

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

    const formatTrendName = (trend) => {
        return trend.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    };

    return (
        <Paper sx={{ p: 3, mt: 3, background: 'linear-gradient(135deg, #1a237e 0%, #283593 100%)', color: 'white' }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                🎯 Combined Model Predictions
            </Typography>
            <Typography variant="subtitle2" gutterBottom>
                Total Models: {combinedPredictions.total_models}
            </Typography>
            <Box sx={{ mt: 2 }}>
                <Grid container spacing={2}>
                    {combinedPredictions.trends.map((trend, index) => (
                        <Grid item xs={12} key={index}>
                            <Paper sx={{ 
                                p: 2, 
                                background: trend.trend === combinedPredictions.dominant_trend 
                                    ? 'linear-gradient(135deg, #00897b 0%, #00695c 100%)'
                                    : 'rgba(255,255,255,0.1)',
                                borderRadius: 2
                            }}>
                                <Grid container alignItems="center" spacing={2}>
                                    <Grid item xs={12} sm={3}>
                                        <Typography variant="subtitle1" sx={{ 
                                            display: 'flex', 
                                            alignItems: 'center', 
                                            gap: 1,
                                            fontWeight: trend.trend === combinedPredictions.dominant_trend ? 'bold' : 'normal'
                                        }}>
                                            {getTrendEmoji(trend.trend)} {formatTrendName(trend.trend)}
                                        </Typography>
                                    </Grid>
                                    <Grid item xs={12} sm={3}>
                                        <Typography variant="body2">
                                            Models: {trend.count} ({trend.percentage.toFixed(1)}%)
                                        </Typography>
                                    </Grid>
                                    <Grid item xs={12} sm={3}>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Typography variant="body2">Confidence:</Typography>
                                            <LinearProgress
                                                variant="determinate"
                                                value={trend.confidence * 100}
                                                sx={{ 
                                                    width: '100%',
                                                    height: 8,
                                                    borderRadius: 4,
                                                    backgroundColor: 'rgba(255,255,255,0.2)',
                                                    '& .MuiLinearProgress-bar': {
                                                        borderRadius: 4,
                                                        background: 'linear-gradient(45deg, #00e676 30%, #69f0ae 90%)',
                                                    }
                                                }}
                                            />
                                            <Typography variant="body2">
                                                {(trend.confidence * 100).toFixed(1)}%
                                            </Typography>
                                        </Box>
                                    </Grid>
                                    <Grid item xs={12} sm={3}>
                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                            {trend.models.slice(0, 3).map((model, idx) => (
                                                <Chip
                                                    key={idx}
                                                    label={`${model.emoji} ${model.name}`}
                                                    size="small"
                                                    sx={{ 
                                                        backgroundColor: 'rgba(255,255,255,0.1)',
                                                        color: 'white',
                                                        '&:hover': {
                                                            backgroundColor: 'rgba(255,255,255,0.2)',
                                                        }
                                                    }}
                                                />
                                            ))}
                                            {trend.models.length > 3 && (
                                                <Chip
                                                    label={`+${trend.models.length - 3} more`}
                                                    size="small"
                                                    sx={{ 
                                                        backgroundColor: 'rgba(255,255,255,0.1)',
                                                        color: 'white'
                                                    }}
                                                />
                                            )}
                                        </Box>
                                    </Grid>
                                </Grid>
                            </Paper>
                        </Grid>
                    ))}
                </Grid>
            </Box>
        </Paper>
    );
};

const AutoMLAnalysis = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
        setError(null);
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a file first');
            return;
        }

        setLoading(true);
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('http://localhost:8000/api/automl/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            setResults(data);
            setError(null);
        } catch (err) {
            setError('Failed to analyze data: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" gutterBottom sx={{ 
                background: 'linear-gradient(45deg, #1a237e 30%, #283593 90%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 3
            }}>
                🤖 AutoML Market Analysis
            </Typography>

            <Paper sx={{ p: 3, mb: 3 }}>
                <Grid container spacing={2} alignItems="center">
                    <Grid item>
                        <Button
                            component="label"
                            variant="contained"
                            startIcon={<CloudUploadIcon />}
                            sx={{
                                background: 'linear-gradient(45deg, #1a237e 30%, #283593 90%)',
                                color: 'white'
                            }}
                        >
                            Upload Data
                            <VisuallyHiddenInput type="file" onChange={handleFileChange} accept=".csv" />
                        </Button>
                    </Grid>
                    <Grid item>
                        {selectedFile && (
                            <Button
                                onClick={handleUpload}
                                variant="contained"
                                startIcon={<AutoGraphIcon />}
                                disabled={loading}
                                sx={{
                                    background: 'linear-gradient(45deg, #00c853 30%, #69f0ae 90%)',
                                    color: 'white'
                                }}
                            >
                                Analyze
                            </Button>
                        )}
                    </Grid>
                    <Grid item>
                        {selectedFile && (
                            <Chip
                                label={selectedFile.name}
                                onDelete={() => setSelectedFile(null)}
                                sx={{ ml: 1 }}
                            />
                        )}
                    </Grid>
                </Grid>

                {error && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                        {error}
                    </Alert>
                )}

                {loading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                        <CircularProgress />
                    </Box>
                )}

                {results && (
                    <Fade in={true}>
                        <Box sx={{ mt: 3 }}>
                            <ConsensusPredictionCard prediction={results.consensus_prediction} />
                            
                            <CombinedPredictionsTable combinedPredictions={results.combined_predictions} />
                            
                            <PressureAnalysisCard pressure={results.pressure_analysis} />
                            
                            <Typography variant="h6" gutterBottom sx={{ mt: 4, mb: 3 }}>
                                🤖 Model Predictions
                            </Typography>
                            
                            <ModelGrid models={Array.isArray(results.models) ? results.models : Object.values(results.models)} />

                            <Paper sx={{ p: 3, mt: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    🎯 Feature Importance
                                </Typography>
                                <TableContainer>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Feature</TableCell>
                                                <TableCell>Importance Score</TableCell>
                                                <TableCell>Impact Level</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {results.feature_importance.map((feature, index) => (
                                                <TableRow key={index}>
                                                    <TableCell>{feature.name}</TableCell>
                                                    <TableCell>
                                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                            <LinearProgress
                                                                variant="determinate"
                                                                value={feature.importance * 100}
                                                                sx={{
                                                                    width: 100,
                                                                    height: 10,
                                                                    borderRadius: 5
                                                                }}
                                                            />
                                                            <Typography>
                                                                {(feature.importance * 100).toFixed(2)}%
                                                            </Typography>
                                                        </Box>
                                                    </TableCell>
                                                    <TableCell>
                                                        {feature.importance > 0.7 ? '🔥 High' :
                                                         feature.importance > 0.3 ? '📈 Medium' : '⚡ Low'}
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </Paper>
                        </Box>
                    </Fade>
                )}
            </Paper>
        </Box>
    );
};

export default AutoMLAnalysis;
