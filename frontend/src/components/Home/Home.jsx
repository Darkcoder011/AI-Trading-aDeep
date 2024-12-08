import React from 'react';
import { Box, Typography, Container, Grid, Paper } from '@mui/material';
import { motion } from 'framer-motion';

const Home = () => {
    return (
        <Container maxWidth="xl">
            <Box sx={{ 
                py: 4, 
                color: 'white',
                textAlign: 'center'
            }}>
                <motion.div
                    initial={{ opacity: 0, y: 50 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                >
                    <Typography 
                        variant="h3" 
                        gutterBottom 
                        sx={{ 
                            fontWeight: 'bold', 
                            color: '#4a6cf7',
                            textShadow: '0 0 10px rgba(74, 108, 247, 0.5)'
                        }}
                    >
                        Welcome to ZyraTech AI Trading Platform
                    </Typography>
                </motion.div>

                <Grid container spacing={3} justifyContent="center" sx={{ mt: 4 }}>
                    {[
                        { 
                            title: 'Trading Dashboard', 
                            description: 'Comprehensive market insights and trading tools',
                            color: '#3a5fff'
                        },
                        { 
                            title: 'AI Analysis', 
                            description: 'Advanced AI-powered market predictions',
                            color: '#4a6cf7'
                        },
                        { 
                            title: 'Real-time Monitoring', 
                            description: 'Live market trends and indicators',
                            color: '#5a7cff'
                        }
                    ].map((feature, index) => (
                        <Grid item xs={12} md={4} key={index}>
                            <motion.div
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                <Paper 
                                    sx={{ 
                                        p: 3, 
                                        background: 'rgba(58, 95, 255, 0.1)',
                                        border: `2px solid ${feature.color}`,
                                        borderRadius: 3,
                                        transition: 'all 0.3s ease',
                                        '&:hover': {
                                            background: 'rgba(58, 95, 255, 0.2)',
                                            transform: 'translateY(-10px)'
                                        }
                                    }}
                                >
                                    <Typography 
                                        variant="h5" 
                                        sx={{ 
                                            color: feature.color, 
                                            fontWeight: 'bold',
                                            mb: 2
                                        }}
                                    >
                                        {feature.title}
                                    </Typography>
                                    <Typography 
                                        variant="body1" 
                                        sx={{ 
                                            color: 'rgba(255,255,255,0.8)' 
                                        }}
                                    >
                                        {feature.description}
                                    </Typography>
                                </Paper>
                            </motion.div>
                        </Grid>
                    ))}
                </Grid>
            </Box>
        </Container>
    );
};

export default Home;
