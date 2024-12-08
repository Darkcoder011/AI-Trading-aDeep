import React from 'react';
import { Grid, Typography } from '@mui/material';
import { ResponsiveContainer, ComposedChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts';

const TechnicalIndicators = ({ data }) => {
  if (!data || !data.technical_indicators) return null;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>Technical Indicators</Typography>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={data.technical_indicators}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="datetime" 
              tick={{ fontSize: 12 }}
              angle={-45}
              textAnchor="end"
            />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Legend />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="close"
              stroke="#8884d8"
              name="Price"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="rsi"
              stroke="#82ca9d"
              name="RSI"
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="macd"
              stroke="#ffc107"
              name="MACD"
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="bb_upper"
              stroke="#dc3545"
              name="BB Upper"
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="bb_lower"
              stroke="#dc3545"
              name="BB Lower"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </Grid>
    </Grid>
  );
};

export default TechnicalIndicators;
