import React, { useState } from 'react';
import { 
  Grid, 
  Paper, 
  Typography, 
  TableContainer, 
  Table, 
  TableHead, 
  TableBody, 
  TableRow, 
  TableCell,
  Box,
  LinearProgress,
  Button,
  alpha,
  Container,
  Input
} from '@mui/material';
import { motion } from 'framer-motion';
import axios from 'axios';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const ModelResults = () => {
  const [file, setFile] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file first');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/analyze_models', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setModelResults(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file. Please try again.');
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ py: 4, color: 'white', textAlign: 'center' }}>
        <Typography 
          variant="h4" 
          gutterBottom 
          sx={{ 
            fontWeight: 'bold', 
            color: '#4a6cf7',
            textShadow: '0 0 10px rgba(74, 108, 247, 0.5)'
          }}
        >
          Model Analysis
        </Typography>

        <Grid container spacing={3} justifyContent="center" sx={{ mb: 4 }}>
          <Grid item xs={12} md={6}>
            <Paper 
              sx={{ 
                p: 3, 
                background: 'rgba(58, 95, 255, 0.1)',
                border: '2px solid #3a5fff',
                borderRadius: 3,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
              }}
            >
              <Input 
                type="file" 
                inputProps={{ accept: '.csv' }}
                onChange={handleFileChange}
                sx={{ display: 'none' }}
                id="file-upload"
              />
              <label htmlFor="file-upload">
                <Button 
                  component="span" 
                  variant="contained" 
                  startIcon={<CloudUploadIcon />}
                  sx={{ 
                    mb: 2, 
                    background: 'linear-gradient(45deg, #3a5fff 30%, #4a6cf7 90%)',
                    '&:hover': {
                      background: 'linear-gradient(45deg, #4a6cf7 30%, #5a7cff 90%)'
                    }
                  }}
                >
                  Upload CSV File
                </Button>
              </label>
              {file && (
                <Typography variant="body2" sx={{ color: 'white', opacity: 0.7 }}>
                  Selected: {file.name}
                </Typography>
              )}
              <Button 
                variant="outlined" 
                onClick={handleUpload} 
                disabled={!file || loading}
                sx={{ 
                  mt: 2, 
                  color: '#4a6cf7', 
                  borderColor: '#4a6cf7',
                  '&:hover': {
                    borderColor: '#3a5fff',
                    backgroundColor: 'rgba(58, 95, 255, 0.1)'
                  }
                }}
              >
                {loading ? 'Analyzing...' : 'Analyze Models'}
              </Button>
            </Paper>
          </Grid>
        </Grid>

        {loading && (
          <LinearProgress 
            sx={{ 
              my: 3, 
              backgroundColor: 'rgba(58, 95, 255, 0.2)', 
              '& .MuiLinearProgress-bar': { 
                backgroundColor: '#4a6cf7' 
              } 
            }} 
          />
        )}

        {modelResults && (
          <>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Paper 
                  elevation={3} 
                  sx={{ 
                    p: 3, 
                    bgcolor: 'background.default',
                    border: 3,
                    borderColor: theme => {
                      const decision = modelResults?.ensemble_decision?.decision;
                      if (decision?.includes('STRONG_BUY') || decision?.includes('BUY')) return theme.palette.success.main;
                      if (decision?.includes('STRONG_SELL') || decision?.includes('SELL')) return theme.palette.error.main;
                      return theme.palette.warning.main;
                    },
                    borderRadius: 2
                  }}
                >
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Box sx={{ textAlign: 'center', mb: 2 }}>
                        <Typography variant="h6" gutterBottom>
                          Ensemble Trading Signal
                        </Typography>
                        <Typography 
                          variant="h3" 
                          sx={{ 
                            fontWeight: 'bold',
                            color: theme => {
                              const decision = modelResults?.ensemble_decision?.decision;
                              if (decision?.includes('STRONG_BUY') || decision?.includes('BUY')) return theme.palette.success.main;
                              if (decision?.includes('STRONG_SELL') || decision?.includes('SELL')) return theme.palette.error.main;
                              return theme.palette.warning.main;
                            }
                          }}
                        >
                          {modelResults?.ensemble_decision?.decision}
                        </Typography>
                        <Typography variant="h6" sx={{ mt: 1, opacity: 0.8 }}>
                          Confidence Level: {modelResults?.ensemble_decision?.confidence_level?.toFixed(1)}%
                        </Typography>
                      </Box>
                    </Grid>

                    <Grid item xs={6}>
                      <Box sx={{ p: 2, bgcolor: 'success.main', borderRadius: 2, color: 'white' }}>
                        <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center' }}>
                          UP Probability
                        </Typography>
                        <Box sx={{ mt: 1, mb: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={modelResults?.ensemble_decision?.up_probability * 100 || 0}
                            sx={{
                              height: 12,
                              borderRadius: 6,
                              backgroundColor: 'rgba(255,255,255,0.3)',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: 'white',
                                borderRadius: 6
                              }
                            }}
                          />
                        </Box>
                        <Typography variant="h6" align="right">
                          {(modelResults?.ensemble_decision?.up_probability * 100)?.toFixed(1)}% 
                        </Typography>
                      </Box>
                    </Grid>

                    <Grid item xs={6}>
                      <Box sx={{ p: 2, bgcolor: 'error.main', borderRadius: 2, color: 'white' }}>
                        <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center' }}>
                          DOWN Probability
                        </Typography>
                        <Box sx={{ mt: 1, mb: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={modelResults?.ensemble_decision?.down_probability * 100 || 0}
                            sx={{
                              height: 12,
                              borderRadius: 6,
                              backgroundColor: 'rgba(255,255,255,0.3)',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: 'white',
                                borderRadius: 6
                              }
                            }}
                          />
                        </Box>
                        <Typography variant="h6" align="right">
                          {(modelResults?.ensemble_decision?.down_probability * 100)?.toFixed(1)}% 
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>

              <Grid item xs={12}>
                <Paper elevation={3} sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    Individual Model Predictions
                  </Typography>
                  {modelResults?.model_predictions ? (
                    <TableContainer>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Model</TableCell>
                            <TableCell align="right">Accuracy </TableCell>
                            <TableCell align="right">F1 Score </TableCell>
                            <TableCell align="right">UP Probability </TableCell>
                            <TableCell align="right">DOWN Probability </TableCell>
                            <TableCell align="right">Weight </TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(modelResults.model_predictions)
                            .sort((a, b) => (b[1].accuracy + b[1].f1)/2 - (a[1].accuracy + a[1].f1)/2)
                            .map(([name, metrics]) => (
                              <TableRow 
                                key={name}
                                sx={{
                                  '&:hover': {
                                    backgroundColor: 'action.hover',
                                  },
                                  backgroundColor: theme => 
                                    metrics?.accuracy > 0.7 ? alpha(theme.palette.success.light, 0.1) : 'inherit'
                                }}
                              >
                                <TableCell sx={{ fontWeight: 'bold' }}>{name}</TableCell>
                                <TableCell align="right">
                                  {(metrics?.accuracy * 100)?.toFixed(2)}%
                                </TableCell>
                                <TableCell align="right">
                                  {(metrics?.f1 * 100)?.toFixed(2)}%
                                </TableCell>
                                <TableCell align="right" sx={{
                                  color: theme => metrics?.probability?.up > 0.6 ? theme.palette.success.main : 'inherit',
                                  fontWeight: metrics?.probability?.up > 0.6 ? 'bold' : 'normal'
                                }}>
                                  {(metrics?.probability?.up * 100)?.toFixed(1)}%
                                </TableCell>
                                <TableCell align="right" sx={{
                                  color: theme => metrics?.probability?.down > 0.6 ? theme.palette.error.main : 'inherit',
                                  fontWeight: metrics?.probability?.down > 0.6 ? 'bold' : 'normal'
                                }}>
                                  {(metrics?.probability?.down * 100)?.toFixed(1)}%
                                </TableCell>
                                <TableCell align="right">
                                  {(modelResults?.model_weights?.[name] * 100)?.toFixed(1)}%
                                </TableCell>
                              </TableRow>
                            ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  ) : (
                    <Typography variant="body1" sx={{ textAlign: 'center', py: 3 }}>
                      No model predictions available
                    </Typography>
                  )}
                </Paper>
              </Grid>

              <Grid item xs={12}>
                <Paper elevation={3} sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    Model Analysis
                  </Typography>
                  {modelResults?.model_analysis ? (
                    <Box sx={{ p: 2 }}>
                      <Typography variant="body1">
                        {modelResults.model_analysis}
                      </Typography>
                    </Box>
                  ) : (
                    <Typography variant="body1" sx={{ textAlign: 'center', py: 3 }}>
                      No model analysis available
                    </Typography>
                  )}
                </Paper>
              </Grid>

              <Grid item xs={12}>
                <Paper elevation={3} sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    Confidence Analysis
                  </Typography>
                  {modelResults?.confidence_analysis ? (
                    <Box sx={{ p: 2 }}>
                      <Typography variant="body1">
                        {modelResults.confidence_analysis}
                      </Typography>
                    </Box>
                  ) : (
                    <Typography variant="body1" sx={{ textAlign: 'center', py: 3 }}>
                      No confidence analysis available
                    </Typography>
                  )}
                </Paper>
              </Grid>
            </Grid>
          </>
        )}
      </Box>
    </Container>
  );
};

export default ModelResults;
