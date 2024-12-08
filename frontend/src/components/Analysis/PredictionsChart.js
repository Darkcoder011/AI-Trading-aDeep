import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Fade,
  Grow,
  Zoom,
  Card,
  CardContent,
  useTheme,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import { styled, keyframes } from '@mui/material/styles';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import RefreshIcon from '@mui/icons-material/Refresh';

// Styled components
const StyledCard = styled(Card)(({ theme }) => ({
  background: 'rgba(255, 255, 255, 0.9)',
  backdropFilter: 'blur(10px)',
  borderRadius: theme.spacing(2),
  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
  border: '1px solid rgba(255, 255, 255, 0.18)',
  transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-5px)',
    boxShadow: '0 12px 40px 0 rgba(31, 38, 135, 0.45)',
  },
}));

const pulseAnimation = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

const fadeInUp = keyframes`
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

const glowAnimation = keyframes`
  0% { box-shadow: 0 0 5px rgba(0,0,0,0.1); }
  50% { box-shadow: 0 0 20px rgba(0,0,0,0.2); }
  100% { box-shadow: 0 0 5px rgba(0,0,0,0.1); }
`;

const PulsatingBox = styled(Box)(({ theme }) => ({
  animation: `${pulseAnimation} 2s infinite ease-in-out`,
}));

const AnimatedBox = styled(Box)`
  animation: ${fadeInUp} 0.5s ease-out;
`;

const GlowingCard = styled(StyledCard)`
  animation: ${glowAnimation} 3s infinite ease-in-out;
`;

const PulsingTypography = styled(Typography)`
  animation: ${pulseAnimation} 2s infinite ease-in-out;
`;

const StyledTableContainer = styled(TableContainer)`
  &::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  &::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
  }
  &::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
    &:hover {
      background: #555;
    }
  }
`;

const getMarketStateEmoji = (state) => {
  switch (state?.toLowerCase()) {
    case 'uptrend':
      return '🚀';
    case 'downtrend':
      return '🔻';
    case 'sideways':
      return '↔️';
    default:
      return '📊';
  }
};

const getConfidenceColor = (score) => {
  if (score >= 0.7) return '#4caf50';
  if (score >= 0.5) return '#ff9800';
  return '#f44336';
};

const getStrategyEmoji = (signal) => {
  switch(signal) {
    case 'strong_bullish':
      return '🚀';
    case 'bullish':
      return '🟢';
    case 'strong_bearish':
      return '💥';
    case 'bearish':
      return '🔴';
    case 'neutral':
      return '⚪';
    default:
      return '❓';
  }
};

const getOverallSentimentEmoji = (sentiment) => {
  switch(sentiment) {
    case 'strong_uptrend':
      return '🚀';
    case 'uptrend':
      return '📈';
    case 'strong_downtrend':
      return '🔻';
    case 'downtrend':
      return '📉';
    case 'sideways':
      return '↔️';
    default:
      return '❓';
  }
};

const getStrategyDescription = (strategy) => {
  const descriptions = {
    "RSI + MACD": "Combines momentum and trend following 📊",
    "Bollinger + Stochastic": "Volatility and momentum strategy 📈",
    "ADX + DMI": "Trend strength and direction analysis 📉",
    "Volume + MFI": "Volume-based momentum strategy 📊",
    "Triple Screen": "Multi-timeframe analysis system 🎯",
    "Trend Following": "Pure trend-following strategy 📈",
    "Momentum + Volume": "Volume-confirmed momentum 📊",
    "Support Resistance": "Key level breakout strategy 🎯",
    "Volatility Breakout": "Volatility expansion system 📈",
    "Price Action": "Pure price movement analysis 📊"
  };
  return descriptions[strategy] || strategy;
};

const getConfidenceEmoji = (confidence) => {
  if (confidence >= 0.8) return '🎯';
  if (confidence >= 0.6) return '✨';
  if (confidence >= 0.4) return '👍';
  return '🤔';
};

const PredictionsChart = ({ predictions, isLoading, onAnalyze }) => {
  const theme = useTheme();

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (!predictions) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, p: 4 }}>
        <Typography variant="h6" color="text.secondary">
          No predictions available
        </Typography>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={onAnalyze}
          startIcon={<RefreshIcon />}
        >
          Start Analysis
        </Button>
      </Box>
    );
  }

  const {
    next_15min_prediction,
    technical_indicators,
    volume_analysis,
    market_strength
  } = predictions;

  return (
    <Box sx={{ p: 3 }}>
      <Fade in timeout={1000}>
        <Typography variant="h4" gutterBottom sx={{ mb: 4, textAlign: 'center' }}>
          Market Predictions {getMarketStateEmoji(next_15min_prediction?.market_state)}
        </Typography>
      </Fade>

      <Grid container spacing={3}>
        {/* Prediction Card */}
        <Grid item xs={12} md={6}>
          <Grow in timeout={1000}>
            <GlowingCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  15-Minute Forecast 🎯
                </Typography>
                <PulsatingBox>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <TrendingUpIcon sx={{ color: '#4caf50', mr: 1 }} />
                    <Typography variant="body1">
                      Upward: {(next_15min_prediction?.direction_probability?.upward * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <TrendingDownIcon sx={{ color: '#f44336', mr: 1 }} />
                    <Typography variant="body1">
                      Downward: {(next_15min_prediction?.direction_probability?.downward * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </PulsatingBox>
                <Box sx={{ 
                  mt: 2, 
                  p: 2, 
                  bgcolor: 'rgba(0,0,0,0.03)', 
                  borderRadius: 2,
                  border: '1px solid rgba(0,0,0,0.1)'
                }}>
                  <Typography variant="subtitle1">
                    Market State: {next_15min_prediction?.market_state} {getMarketStateEmoji(next_15min_prediction?.market_state)}
                  </Typography>
                  <Typography variant="subtitle1">
                    Confidence: {(next_15min_prediction?.confidence_score * 100).toFixed(1)}% 
                    <span style={{ 
                      color: getConfidenceColor(next_15min_prediction?.confidence_score),
                      marginLeft: '8px'
                    }}>
                      {next_15min_prediction?.confidence_score >= 0.7 ? '🎯' : 
                       next_15min_prediction?.confidence_score >= 0.5 ? '🎲' : '⚠️'}
                    </span>
                  </Typography>
                </Box>
              </CardContent>
            </GlowingCard>
          </Grow>
        </Grid>

        {/* Technical Indicators Card */}
        <Grid item xs={12} md={6}>
          <Zoom in timeout={1500}>
            <GlowingCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Technical Analysis 📈
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">RSI</Typography>
                    <Typography variant="body1">{technical_indicators?.rsi?.toFixed(2)} 
                      {technical_indicators?.rsi > 70 ? '🔥' : 
                       technical_indicators?.rsi < 30 ? '❄️' : ''}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">ADX</Typography>
                    <Typography variant="body1">{technical_indicators?.adx?.toFixed(2)}
                      {technical_indicators?.adx > 25 ? '💪' : ''}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">CCI</Typography>
                    <Typography variant="body1">{technical_indicators?.cci?.toFixed(2)}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="textSecondary">Stochastic</Typography>
                    <Typography variant="body1">{technical_indicators?.stochastic?.toFixed(2)}</Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </GlowingCard>
          </Zoom>
        </Grid>

        {/* Volume Analysis Card */}
        <Grid item xs={12} md={6}>
          <Grow in timeout={2000}>
            <GlowingCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Volume Analysis 📊
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="textSecondary">Volume Trend</Typography>
                  <Typography variant="body1">
                    {volume_analysis?.volume_trend} 
                    {volume_analysis?.volume_trend === 'increasing' ? ' 📈' : ' 📉'}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">Volume Strength</Typography>
                  <Typography variant="body1">
                    {(volume_analysis?.volume_strength * 100).toFixed(1)}%
                    {volume_analysis?.volume_strength > 1.5 ? ' 💪' : 
                     volume_analysis?.volume_strength < 0.5 ? ' 😴' : ' 👍'}
                  </Typography>
                </Box>
              </CardContent>
            </GlowingCard>
          </Grow>
        </Grid>

        {/* Market Strength Card */}
        <Grid item xs={12} md={6}>
          <Zoom in timeout={2500}>
            <GlowingCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Market Strength 💪
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="textSecondary">Trend Strength</Typography>
                  <Typography variant="body1">
                    {market_strength?.trend_strength?.toFixed(2)}
                    {market_strength?.trend_strength > 25 ? ' 🚀' : ' ⚖️'}
                  </Typography>
                </Box>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="textSecondary">Volatility</Typography>
                  <Typography variant="body1">
                    {market_strength?.volatility?.toFixed(2)}
                    {market_strength?.volatility > 0.02 ? ' 🌋' : ' 🌊'}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">Momentum</Typography>
                  <Typography variant="body1">
                    {market_strength?.momentum?.toFixed(2)}
                    {market_strength?.momentum > 70 ? ' 🏃' : 
                     market_strength?.momentum < 30 ? ' 🐌' : ' 🚶'}
                  </Typography>
                </Box>
              </CardContent>
            </GlowingCard>
          </Zoom>
        </Grid>

        {/* Strategy Analysis Table */}
        <Grid item xs={12}>
          <Zoom in timeout={2000}>
            <GlowingCard>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Trading Strategies Analysis 📊
                </Typography>
                
                {/* Overall Strategy Summary */}
                <AnimatedBox sx={{ mb: 3, p: 2, bgcolor: 'rgba(0,0,0,0.03)', borderRadius: 2 }}>
                  <PulsingTypography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    Market Pulse 💫 {getOverallSentimentEmoji(predictions.strategy_analysis.overall_sentiment)}
                  </PulsingTypography>
                  <Box sx={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                    gap: 2,
                    mt: 2 
                  }}>
                    <Paper elevation={3} sx={{ p: 2, textAlign: 'center', bgcolor: 'rgba(255,255,255,0.9)' }}>
                      <Typography variant="h6">
                        Sentiment Score
                      </Typography>
                      <Typography variant="h4" sx={{ color: theme.palette.primary.main }}>
                        {(predictions.strategy_analysis.sentiment_score * 100).toFixed(1)}%
                      </Typography>
                    </Paper>
                    <Paper elevation={3} sx={{ p: 2, textAlign: 'center', bgcolor: 'rgba(255,255,255,0.9)' }}>
                      <Typography variant="h6">
                        Signal Distribution
                      </Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'space-around', mt: 1 }}>
                        <Box>
                          <Typography variant="body2">Strong Bullish</Typography>
                          <Typography variant="h6">
                            {predictions.strategy_analysis.summary.strong_bullish} 🚀
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="body2">Strong Bearish</Typography>
                          <Typography variant="h6">
                            {predictions.strategy_analysis.summary.strong_bearish} 💥
                          </Typography>
                        </Box>
                      </Box>
                    </Paper>
                  </Box>
                  <Box sx={{ 
                    display: 'flex', 
                    justifyContent: 'space-around', 
                    mt: 2,
                    flexWrap: 'wrap',
                    gap: 2
                  }}>
                    <Chip 
                      icon={<span>🟢</span>}
                      label={`Bullish: ${predictions.strategy_analysis.summary.bullish}`}
                      color="success"
                      variant="outlined"
                    />
                    <Chip 
                      icon={<span>⚪</span>}
                      label={`Neutral: ${predictions.strategy_analysis.summary.neutral}`}
                      color="default"
                      variant="outlined"
                    />
                    <Chip 
                      icon={<span>🔴</span>}
                      label={`Bearish: ${predictions.strategy_analysis.summary.bearish}`}
                      color="error"
                      variant="outlined"
                    />
                  </Box>
                </AnimatedBox>

                {/* Strategies Table */}
                <StyledTableContainer component={Paper} sx={{ maxHeight: 440, overflow: 'auto' }}>
                  <Table stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 'bold', bgcolor: theme.palette.primary.main, color: 'white' }}>
                          Strategy 📈
                        </TableCell>
                        <TableCell align="center" sx={{ fontWeight: 'bold', bgcolor: theme.palette.primary.main, color: 'white' }}>
                          Signal 🎯
                        </TableCell>
                        <TableCell align="center" sx={{ fontWeight: 'bold', bgcolor: theme.palette.primary.main, color: 'white' }}>
                          ML Confidence ✨
                        </TableCell>
                        <TableCell align="center" sx={{ fontWeight: 'bold', bgcolor: theme.palette.primary.main, color: 'white' }}>
                          Up Prob. 📈
                        </TableCell>
                        <TableCell align="center" sx={{ fontWeight: 'bold', bgcolor: theme.palette.primary.main, color: 'white' }}>
                          Down Prob. 📉
                        </TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold', bgcolor: theme.palette.primary.main, color: 'white' }}>
                          Status 🎯
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(predictions.strategy_analysis.strategies).map(([strategy, data], index) => (
                        <Zoom in timeout={300 + index * 100}>
                          <TableRow
                            key={strategy}
                            sx={{
                              '&:last-child td, &:last-child th': { border: 0 },
                              bgcolor: data.signal === 'strong_bullish' 
                                ? 'rgba(76, 175, 80, 0.2)'
                                : data.signal === 'bullish'
                                  ? 'rgba(76, 175, 80, 0.1)'
                                  : data.signal === 'strong_bearish'
                                    ? 'rgba(244, 67, 54, 0.2)'
                                    : data.signal === 'bearish'
                                      ? 'rgba(244, 67, 54, 0.1)'
                                      : 'inherit',
                              transition: 'all 0.3s ease',
                              '&:hover': {
                                bgcolor: data.signal === 'strong_bullish' 
                                  ? 'rgba(76, 175, 80, 0.3)'
                                  : data.signal === 'bullish'
                                    ? 'rgba(76, 175, 80, 0.2)'
                                    : data.signal === 'strong_bearish'
                                      ? 'rgba(244, 67, 54, 0.3)'
                                      : data.signal === 'bearish'
                                        ? 'rgba(244, 67, 54, 0.2)'
                                        : 'rgba(0, 0, 0, 0.04)'
                              }
                            }}
                          >
                            <TableCell component="th" scope="row">
                              <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                                {strategy}
                              </Typography>
                              <Typography variant="caption" color="textSecondary">
                                {getStrategyDescription(strategy)}
                              </Typography>
                            </TableCell>
                            <TableCell align="center">
                              <Chip
                                label={data.signal.split('_').map(word => 
                                  word.charAt(0).toUpperCase() + word.slice(1)
                                ).join(' ')}
                                color={
                                  data.signal.includes('bullish') ? 'success' :
                                  data.signal.includes('bearish') ? 'error' :
                                  'default'
                                }
                                size="small"
                              />
                            </TableCell>
                            <TableCell align="center">
                              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                                <Typography>
                                  {data.ml_confidence 
                                    ? `${(data.ml_confidence * 100).toFixed(1)}%`
                                    : 'N/A'}
                                </Typography>
                                {data.ml_confidence && 
                                  <Typography>{getConfidenceEmoji(data.ml_confidence)}</Typography>
                                }
                              </Box>
                            </TableCell>
                            <TableCell align="center">
                              <Typography sx={{ 
                                color: data.ml_upward_prob > 0.5 ? 'success.main' : 'text.primary',
                                fontWeight: data.ml_upward_prob > 0.7 ? 'bold' : 'normal'
                              }}>
                                {data.ml_upward_prob 
                                  ? `${(data.ml_upward_prob * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </Typography>
                            </TableCell>
                            <TableCell align="center">
                              <Typography sx={{ 
                                color: data.ml_downward_prob > 0.5 ? 'error.main' : 'text.primary',
                                fontWeight: data.ml_downward_prob > 0.7 ? 'bold' : 'normal'
                              }}>
                                {data.ml_downward_prob 
                                  ? `${(data.ml_downward_prob * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              {getStrategyEmoji(data.signal)}
                            </TableCell>
                          </TableRow>
                        </Zoom>
                      ))}
                    </TableBody>
                  </Table>
                </StyledTableContainer>
              </CardContent>
            </GlowingCard>
          </Zoom>
        </Grid>
      </Grid>

      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={onAnalyze}
          startIcon={<RefreshIcon />}
        >
          Refresh Analysis
        </Button>
      </Box>
    </Box>
  );
};

export default PredictionsChart;
