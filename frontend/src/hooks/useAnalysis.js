import { useState } from 'react';
import { uploadFile, analyzeData, analyzeModels } from '../services/api';

export const useAnalysis = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith('.csv')) {
        setError('Please upload a CSV file');
        return;
      }

      setSelectedFile(file);
      setLoading(true);
      setError(null);
      
      try {
        const uploadResponse = await uploadFile(file);
        setChartData(uploadResponse.data);
        
        const analysisResponse = await analyzeData(file);
        setAnalysisResults(analysisResponse);
        
        const modelAnalysis = await analyzeModels(file);
        if (modelAnalysis) {
          setModelResults(modelAnalysis);
        }
      } catch (error) {
        console.error('Error processing file:', error);
        setError(error.response?.data?.detail || 'Error processing file');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please upload a file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await analyzeData(selectedFile);
      setAnalysisResults(response);
      setError(null);
    } catch (error) {
      console.error('Error analyzing data:', error);
      setError(error.response?.data?.detail || 'Error analyzing data. Please try again.');
      setAnalysisResults(null);
    } finally {
      setLoading(false);
    }
  };

  return {
    selectedFile,
    chartData,
    loading,
    analysisResults,
    modelResults,
    error,
    handleFileUpload,
    handleAnalyze,
    setSelectedFile,
    setChartData,
    setLoading,
    setAnalysisResults,
    setModelResults,
    setError,
  };
};

export default useAnalysis;
