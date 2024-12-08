import React, { useState } from 'react';
import { 
    Box, 
    Typography, 
    Container, 
    Grid, 
    Paper, 
    Switch, 
    FormControlLabel, 
    Slider, 
    Select, 
    MenuItem, 
    FormControl, 
    InputLabel 
} from '@mui/material';
import { motion } from 'framer-motion';

const Settings = () => {
    const [darkMode, setDarkMode] = useState(true);
    const [riskTolerance, setRiskTolerance] = useState(50);
    const [analysisPeriod, setAnalysisPeriod] = useState('weekly');

    return (
        <Container maxWidth="md">
            <Box sx={{ 
                py: 4, 
                color: 'white',
                textAlign: 'left'
            }}>
                <Typography 
                    variant="h4" 
                    gutterBottom 
                    sx={{ 
                        fontWeight: 'bold', 
                        color: '#4a6cf7',
                        textShadow: '0 0 10px rgba(74, 108, 247, 0.5)',
                        mb: 4
                    }}
                >
                    Platform Settings
                </Typography>

                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <motion.div
                            whileHover={{ scale: 1.02 }}
                            transition={{ duration: 0.3 }}
                        >
                            <Paper 
                                sx={{ 
                                    p: 3, 
                                    background: 'rgba(58, 95, 255, 0.1)',
                                    border: '2px solid #3a5fff',
                                    borderRadius: 3
                                }}
                            >
                                <Typography variant="h6" sx={{ color: '#4a6cf7', mb: 2 }}>
                                    Appearance
                                </Typography>
                                <FormControlLabel
                                    control={
                                        <Switch 
                                            checked={darkMode}
                                            onChange={() => setDarkMode(!darkMode)}
                                            color="primary"
                                        />
                                    }
                                    label="Dark Mode"
                                    sx={{ color: 'white' }}
                                />
                            </Paper>
                        </motion.div>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <motion.div
                            whileHover={{ scale: 1.02 }}
                            transition={{ duration: 0.3 }}
                        >
                            <Paper 
                                sx={{ 
                                    p: 3, 
                                    background: 'rgba(58, 95, 255, 0.1)',
                                    border: '2px solid #3a5fff',
                                    borderRadius: 3
                                }}
                            >
                                <Typography variant="h6" sx={{ color: '#4a6cf7', mb: 2 }}>
                                    Risk Management
                                </Typography>
                                <Typography sx={{ color: 'white', mb: 1 }}>
                                    Risk Tolerance
                                </Typography>
                                <Slider
                                    value={riskTolerance}
                                    onChange={(e, newValue) => setRiskTolerance(newValue)}
                                    color="primary"
                                    valueLabelDisplay="auto"
                                />
                            </Paper>
                        </motion.div>
                    </Grid>

                    <Grid item xs={12}>
                        <motion.div
                            whileHover={{ scale: 1.02 }}
                            transition={{ duration: 0.3 }}
                        >
                            <Paper 
                                sx={{ 
                                    p: 3, 
                                    background: 'rgba(58, 95, 255, 0.1)',
                                    border: '2px solid #3a5fff',
                                    borderRadius: 3
                                }}
                            >
                                <Typography variant="h6" sx={{ color: '#4a6cf7', mb: 2 }}>
                                    Analysis Preferences
                                </Typography>
                                <FormControl fullWidth>
                                    <InputLabel sx={{ color: 'white' }}>Analysis Period</InputLabel>
                                    <Select
                                        value={analysisPeriod}
                                        label="Analysis Period"
                                        onChange={(e) => setAnalysisPeriod(e.target.value)}
                                        sx={{ 
                                            color: 'white',
                                            '& .MuiOutlinedInput-notchedOutline': {
                                                borderColor: '#3a5fff'
                                            }
                                        }}
                                    >
                                        <MenuItem value="daily">Daily</MenuItem>
                                        <MenuItem value="weekly">Weekly</MenuItem>
                                        <MenuItem value="monthly">Monthly</MenuItem>
                                    </Select>
                                </FormControl>
                            </Paper>
                        </motion.div>
                    </Grid>
                </Grid>
            </Box>
        </Container>
    );
};

export default Settings;
