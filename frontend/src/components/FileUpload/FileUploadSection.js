import React from 'react';
import { Paper, Grid, Button } from '@mui/material';

const FileUploadSection = ({ onFileUpload, onAnalyze, selectedFile, loading }) => {
  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Grid container spacing={2} alignItems="center">
        <Grid item>
          <Button
            variant="contained"
            component="label"
          >
            Upload CSV
            <input
              type="file"
              hidden
              accept=".csv"
              onChange={onFileUpload}
            />
          </Button>
        </Grid>
        <Grid item>
          <Button
            variant="contained"
            onClick={onAnalyze}
            disabled={!selectedFile || loading}
          >
            Analyze Data
          </Button>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default FileUploadSection;
