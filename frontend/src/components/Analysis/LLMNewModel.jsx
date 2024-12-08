import React, { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Button,
    TextField,
    Box,
    Alert,
    Paper,
    CircularProgress,
    Grid,
    Chip,
    Fade,
    Grow,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    TableContainer,
    Table,
    TableHead,
    TableRow,
    TableCell,
    TableBody
} from '@mui/material';
import { styled } from '@mui/material/styles';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import PsychologyIcon from '@mui/icons-material/Psychology';
import axios from 'axios';

const StyledCard = styled(Card)(({ theme }) => ({
    background: 'linear-gradient(135deg, #2c387e 0%, #1976d2 100%)',
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

const ModelResultsTable = ({ predictions, consensus }) => {
    return (
        <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
                🤖 Model Predictions
            </Typography>
            <TableContainer component={Paper}>
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
                        {Object.entries(predictions).map(([modelId, data]) => (
                            <TableRow key={modelId} hover>
                                <TableCell>{data.name}</TableCell>
                                <TableCell>{data.type}</TableCell>
                                <TableCell>
                                    <Chip
                                        label={data.prediction.trend}
                                        color={
                                            data.prediction.trend === 'Uptrend' ? 'success' :
                                            data.prediction.trend === 'Downtrend' ? 'error' : 'warning'
                                        }
                                        size="small"
                                    />
                                </TableCell>
                                <TableCell>
                                    {data.prediction.confidence.toFixed(1)}%
                                </TableCell>
                                <TableCell>
                                    <Typography
                                        color={data.prediction.price_movement >= 0 ? 'success.main' : 'error.main'}
                                    >
                                        {data.prediction.price_movement >= 0 ? '+' : ''}
                                        {data.prediction.price_movement.toFixed(2)}%
                                    </Typography>
                                </TableCell>
                            </TableRow>
                        ))}
                        <TableRow sx={{ bgcolor: 'action.hover' }}>
                            <TableCell colSpan={2}>
                                <Typography variant="subtitle2">Consensus</Typography>
                            </TableCell>
                            <TableCell>
                                <Chip
                                    label={consensus.trend}
                                    color={
                                        consensus.trend === 'Uptrend' ? 'success' :
                                        consensus.trend === 'Downtrend' ? 'error' : 'warning'
                                    }
                                    size="small"
                                />
                            </TableCell>
                            <TableCell>{consensus.confidence.toFixed(1)}%</TableCell>
                            <TableCell>
                                <Typography
                                    color={consensus.price_movement >= 0 ? 'success.main' : 'error.main'}
                                >
                                    {consensus.price_movement >= 0 ? '+' : ''}
                                    {consensus.price_movement.toFixed(2)}%
                                </Typography>
                            </TableCell>
                        </TableRow>
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    );
};

const LLMNewModel = () => {
    const [marketFile, setMarketFile] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleMarketFileUpload = (event) => {
        const file = event.target.files[0];
        if (file && file.type === "text/csv") {
            setMarketFile(file);
            setError(null);
        } else {
            setError("Please upload a valid CSV file");
        }
    };

    const handleAnalyzeMarket = async () => {
        if (!marketFile) {
            setError("Please upload a CSV file");
            return;
        }

        setLoading(true);
        setError(null);
        setAnalysisResult(null);

        const formData = new FormData();
        formData.append('file', marketFile);

        try {
            const response = await axios.post(
                'http://localhost:8000/api/v1/llm-new/analyze-market-all',
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                }
            );

            if (response.data.status === 'success') {
                setAnalysisResult(response.data);
            } else {
                setError('Analysis failed');
            }
        } catch (err) {
            console.error('Error:', err);
            setError(err.response?.data?.detail || 'Error analyzing market data');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box sx={{ p: 3 }}>
            <StyledCard>
                <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                        <PsychologyIcon sx={{ fontSize: 40 }} />
                        <Typography variant="h4" component="div">
                            🤖 Multi-Model Market Analysis
                        </Typography>
                    </Box>

                    <Paper sx={{ p: 3, bgcolor: 'background.paper' }}>
                        <Typography variant="h6" gutterBottom>
                            📊 Upload Market Data
                        </Typography>

                        <AnimatedButton
                            variant="contained"
                            component="label"
                            fullWidth
                            sx={{ mb: 2 }}
                        >
                            Upload CSV (100 Candles)
                            <input
                                type="file"
                                hidden
                                accept=".csv"
                                onChange={handleMarketFileUpload}
                            />
                        </AnimatedButton>

                        {marketFile && (
                            <Typography variant="body2" sx={{ mb: 2, color: 'primary.main' }}>
                                Selected file: {marketFile.name}
                            </Typography>
                        )}

                        <AnimatedButton
                            variant="contained"
                            onClick={handleAnalyzeMarket}
                            disabled={loading || !marketFile}
                            fullWidth
                        >
                            {loading ? (
                                <CircularProgress size={24} color="inherit" />
                            ) : (
                                '🔮 Analyze with All Models'
                            )}
                        </AnimatedButton>
                    </Paper>

                    {analysisResult && (
                        <>
                            <Box sx={{ mt: 3 }}>
                                <Paper sx={{ p: 3 }}>
                                    <Typography variant="h6" gutterBottom>
                                        📈 Market Data
                                    </Typography>
                                    <Grid container spacing={2}>
                                        <Grid item xs={12} sm={6} md={3}>
                                            <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                <Typography variant="body2">Current Price</Typography>
                                                <Typography variant="h6">
                                                    {analysisResult.market_data.current_price.toFixed(2)}
                                                </Typography>
                                            </Paper>
                                        </Grid>
                                        <Grid item xs={12} sm={6} md={3}>
                                            <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                <Typography variant="body2">SMA 20</Typography>
                                                <Typography variant="h6">
                                                    {analysisResult.market_data.technical_indicators.sma_20.toFixed(2)}
                                                </Typography>
                                            </Paper>
                                        </Grid>
                                        <Grid item xs={12} sm={6} md={3}>
                                            <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                <Typography variant="body2">RSI</Typography>
                                                <Typography variant="h6">
                                                    {analysisResult.market_data.technical_indicators.rsi.toFixed(2)}
                                                </Typography>
                                            </Paper>
                                        </Grid>
                                        <Grid item xs={12} sm={6} md={3}>
                                            <Paper sx={{ p: 2, textAlign: 'center' }}>
                                                <Typography variant="body2">MACD</Typography>
                                                <Typography variant="h6">
                                                    {analysisResult.market_data.technical_indicators.macd.toFixed(2)}
                                                </Typography>
                                            </Paper>
                                        </Grid>
                                    </Grid>
                                </Paper>
                            </Box>

                            <ModelResultsTable 
                                predictions={analysisResult.predictions}
                                consensus={analysisResult.consensus}
                            />
                        </>
                    )}

                    {error && (
                        <Alert severity="error" sx={{ mt: 2 }}>
                            {error}
                        </Alert>
                    )}
                </CardContent>
            </StyledCard>
        </Box>
    );
};

export default LLMNewModel;
