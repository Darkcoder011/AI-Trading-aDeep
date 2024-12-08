import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axios.post(`${API_BASE_URL}/upload`, formData);
  return response.data;
};

export const analyzeData = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axios.post(`${API_BASE_URL}/analyze`, formData);
  return response.data;
};

export const analyzeModels = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axios.post(`${API_BASE_URL}/analyze_models`, formData);
  return response.data;
};
