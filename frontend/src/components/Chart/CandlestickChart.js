import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';
import { createChart } from 'lightweight-charts';

const CandlestickChart = ({ data }) => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (chartContainerRef.current && data.length > 0) {
      if (chartRef.current) {
        chartRef.current.remove();
      }

      const chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: 400,
        layout: {
          background: { color: '#ffffff' },
          textColor: '#333',
        },
        grid: {
          vertLines: { color: '#f0f0f0' },
          horzLines: { color: '#f0f0f0' },
        },
      });

      const candlestickSeries = chart.addCandlestickSeries();
      const formattedData = data.map(item => ({
        time: item.timestamp,
        open: parseFloat(item.open),
        high: parseFloat(item.high),
        low: parseFloat(item.low),
        close: parseFloat(item.close),
      }));

      formattedData.sort((a, b) => a.time - b.time);
      candlestickSeries.setData(formattedData);
      chartRef.current = chart;

      const handleResize = () => {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      };

      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }
  }, [data]);

  return <Box ref={chartContainerRef} sx={{ height: 400 }} />;
};

export default CandlestickChart;
