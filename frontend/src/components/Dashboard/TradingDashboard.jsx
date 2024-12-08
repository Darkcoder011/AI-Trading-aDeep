import React, { useState } from 'react';
import { 
    Box, 
    Typography, 
    Container, 
    Grid, 
    Paper, 
    Table, 
    TableBody, 
    TableCell, 
    TableContainer, 
    TableHead, 
    TableRow 
} from '@mui/material';
import { motion } from 'framer-motion';

const TradingDashboard = () => {
    const [marketData, setMarketData] = useState([
        { symbol: 'AAPL', price: 175.23, change: 2.45, changePercent: 1.42 },
        { symbol: 'GOOGL', price: 126.57, change: -1.23, changePercent: -0.96 },
        { symbol: 'MSFT', price: 345.89, change: 3.11, changePercent: 0.91 },
        { symbol: 'AMZN', price: 145.67, change: 1.78, changePercent: 1.23 }
    ]);

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
                        variant="h4" 
                        gutterBottom 
                        sx={{ 
                            fontWeight: 'bold', 
                            color: '#4a6cf7',
                            textShadow: '0 0 10px rgba(74, 108, 247, 0.5)'
                        }}
                    >
                        Trading Dashboard
                    </Typography>
                </motion.div>

                <Grid container spacing={3}>
                    <Grid item xs={12}>
                        <Paper 
                            sx={{ 
                                p: 3, 
                                background: 'rgba(58, 95, 255, 0.1)',
                                border: '2px solid #3a5fff',
                                borderRadius: 3
                            }}
                        >
                            <Typography 
                                variant="h6" 
                                sx={{ 
                                    color: '#4a6cf7', 
                                    mb: 2 
                                }}
                            >
                                Market Overview
                            </Typography>
                            <TableContainer>
                                <Table>
                                    <TableHead>
                                        <TableRow>
                                            <TableCell sx={{ color: 'white' }}>Symbol</TableCell>
                                            <TableCell sx={{ color: 'white' }}>Price</TableCell>
                                            <TableCell sx={{ color: 'white' }}>Change</TableCell>
                                            <TableCell sx={{ color: 'white' }}>Change %</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {marketData.map((stock, index) => (
                                            <TableRow key={index}>
                                                <TableCell sx={{ color: 'white' }}>{stock.symbol}</TableCell>
                                                <TableCell sx={{ color: 'white' }}>${stock.price.toFixed(2)}</TableCell>
                                                <TableCell sx={{ 
                                                    color: stock.change >= 0 ? 'green' : 'red' 
                                                }}>
                                                    {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}
                                                </TableCell>
                                                <TableCell sx={{ 
                                                    color: stock.changePercent >= 0 ? 'green' : 'red' 
                                                }}>
                                                    {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </Paper>
                    </Grid>
                </Grid>
            </Box>
        </Container>
    );
};

export default TradingDashboard;
