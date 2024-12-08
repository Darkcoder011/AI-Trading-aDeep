import React from 'react';
import { Grid, Paper, Typography, Box, alpha } from '@mui/material';

const MarketContext = ({ modelResults }) => {
  if (!modelResults?.market_context) return null;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            📊 Market Context Analysis
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ 
                p: 2, 
                bgcolor: 'background.paper', 
                borderRadius: 2,
                border: 1,
                borderColor: theme => {
                  const trend = modelResults?.market_context?.trend_confirmation;
                  if (trend?.includes('strong_uptrend')) return theme.palette.success.main;
                  if (trend?.includes('uptrend')) return alpha(theme.palette.success.main, 0.5);
                  if (trend?.includes('strong_downtrend')) return theme.palette.error.main;
                  if (trend?.includes('downtrend')) return alpha(theme.palette.error.main, 0.5);
                  return theme.palette.grey[300];
                }
              }}>
                <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <span>📈</span> Trend Confirmation
                </Typography>
                <Typography variant="h6" sx={{ 
                  color: theme => {
                    const trend = modelResults?.market_context?.trend_confirmation;
                    if (trend?.includes('strong_uptrend')) return theme.palette.success.dark;
                    if (trend?.includes('uptrend')) return theme.palette.success.main;
                    if (trend?.includes('strong_downtrend')) return theme.palette.error.dark;
                    if (trend?.includes('downtrend')) return theme.palette.error.main;
                    return theme.palette.text.secondary;
                  },
                  fontWeight: 'bold',
                  textTransform: 'capitalize'
                }}>
                  {modelResults?.market_context?.trend_confirmation?.replace('_', ' ') || 'N/A'}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ 
                p: 2, 
                bgcolor: 'background.paper', 
                borderRadius: 2,
                border: 1,
                borderColor: theme => modelResults?.market_context?.volume_confirmation ? 
                  theme.palette.success.main : theme.palette.grey[300]
              }}>
                <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <span>📊</span> Volume Analysis
                </Typography>
                <Typography variant="h6" sx={{ 
                  color: theme => modelResults?.market_context?.volume_confirmation ? 
                    theme.palette.success.main : theme.palette.text.secondary,
                  fontWeight: 'bold'
                }}>
                  {modelResults?.market_context?.volume_confirmation ? 'Strong Volume' : 'Weak Volume'}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ 
                p: 2, 
                bgcolor: 'background.paper', 
                borderRadius: 2,
                border: 1,
                borderColor: theme => {
                  const signal = modelResults?.market_context?.rsi_signal;
                  if (signal?.includes('strong_buy')) return theme.palette.success.main;
                  if (signal?.includes('buy')) return alpha(theme.palette.success.main, 0.5);
                  if (signal?.includes('strong_sell')) return theme.palette.error.main;
                  if (signal?.includes('sell')) return alpha(theme.palette.error.main, 0.5);
                  return theme.palette.grey[300];
                }
              }}>
                <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <span>🎯</span> RSI Signal
                </Typography>
                <Typography variant="h6" sx={{ 
                  color: theme => {
                    const signal = modelResults?.market_context?.rsi_signal;
                    if (signal?.includes('strong_buy')) return theme.palette.success.dark;
                    if (signal?.includes('buy')) return theme.palette.success.main;
                    if (signal?.includes('strong_sell')) return theme.palette.error.dark;
                    if (signal?.includes('sell')) return theme.palette.error.main;
                    return theme.palette.text.secondary;
                  },
                  fontWeight: 'bold',
                  textTransform: 'capitalize'
                }}>
                  {modelResults?.market_context?.rsi_signal?.replace('_', ' ') || 'Neutral'}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ 
                p: 2, 
                bgcolor: 'background.paper', 
                borderRadius: 2,
                border: 1,
                borderColor: theme => {
                  const vol = modelResults?.market_context?.volatility_adjustment || 0;
                  if (vol > 50) return theme.palette.error.main;
                  if (vol > 30) return theme.palette.warning.main;
                  return theme.palette.success.main;
                }
              }}>
                <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <span>📉</span> Volatility
                </Typography>
                <Typography variant="h6" sx={{ 
                  color: theme => {
                    const vol = modelResults?.market_context?.volatility_adjustment || 0;
                    if (vol > 50) return theme.palette.error.main;
                    if (vol > 30) return theme.palette.warning.main;
                    return theme.palette.success.main;
                  },
                  fontWeight: 'bold'
                }}>
                  {modelResults?.market_context?.volatility_adjustment?.toFixed(1)}%
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            🔍 Market Analysis
          </Typography>
          {modelResults?.market_context?.analysis ? (
            <Box sx={{ p: 2 }}>
              <Typography variant="body1">
                {modelResults.market_context.analysis}
              </Typography>
            </Box>
          ) : (
            <Typography variant="body1" sx={{ textAlign: 'center', py: 3 }}>
              No market analysis available
            </Typography>
          )}
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            📈 Technical Analysis
          </Typography>
          {modelResults?.market_context?.technical_analysis ? (
            <Box sx={{ p: 2 }}>
              <Typography variant="body1">
                {modelResults.market_context.technical_analysis}
              </Typography>
            </Box>
          ) : (
            <Typography variant="body1" sx={{ textAlign: 'center', py: 3 }}>
              No technical analysis available
            </Typography>
          )}
        </Paper>
      </Grid>
    </Grid>
  );
};

export default MarketContext;
