import React, { useState } from 'react';
import { Box, Button, Typography, Paper } from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PredictionsChart from './PredictionsChart';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const MarketAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const analyzeCandlesticks = async () => {
    if (!file) {
      setError('Please select a CSV file first');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze market data');
      }

      const data = await response.json();
      setPredictions(data);
    } catch (err) {
      setError(err.message);
      setPredictions(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
          <Typography variant="h5" gutterBottom>
            Market Analysis Tool
          </Typography>
          
          <Button
            component="label"
            variant="contained"
            startIcon={<CloudUploadIcon />}
            sx={{ mb: 2 }}
          >
            Upload Candlestick Data
            <VisuallyHiddenInput 
              type="file" 
              accept=".csv"
              onChange={handleFileChange}
            />
          </Button>
          
          {file && (
            <Typography variant="body2" color="text.secondary">
              Selected file: {file.name}
            </Typography>
          )}
          
          {error && (
            <Typography variant="body2" color="error" sx={{ mt: 1 }}>
              {error}
            </Typography>
          )}
        </Box>
      </Paper>

      <PredictionsChart 
        predictions={predictions}
        isLoading={isLoading}
        onAnalyze={analyzeCandlesticks}
      />
    </Box>
  );
};

export default MarketAnalyzer;
