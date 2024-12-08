import React, { useState, useMemo, useEffect } from 'react';
import { 
  Routes, 
  Route, 
  Link,
  useNavigate,
  useLocation,
  Navigate
} from 'react-router-dom';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Container, 
  Button, 
  Box, 
  CssBaseline, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText,
  IconButton,
  Switch,
  FormControlLabel
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { motion } from 'framer-motion';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';

// Icons
import MenuIcon from '@mui/icons-material/Menu';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import SettingsIcon from '@mui/icons-material/Settings';
import BrainIcon from '@mui/icons-material/Psychology';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';

// Import components
import Home from './components/Home/Home';
import TradingDashboard from './components/Dashboard/TradingDashboard';
import MarketAnalyzer from './components/Analysis/MarketAnalyzer';
import LLMModelAnalysis from './components/Analysis/LLMModelAnalysis';
import AutoMLAnalysis from './components/Analysis/AutoMLAnalysis';
import AIDashboard from './components/Dashboard/AIDashboard';
import Settings from './components/Settings/Settings';
import ModelResults from './components/Analysis/ModelResults';
import LLMNewModel from './components/Analysis/LLMNewModel';

import { useAuth } from './context/AuthContext';

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, isSecurityVerified } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/', { replace: true });
    } else if (!isSecurityVerified) {
      navigate('/enter-password', { replace: true });
    }
  }, [isAuthenticated, isSecurityVerified, navigate]);

  return isAuthenticated && isSecurityVerified ? children : null;
};

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [mode, setMode] = useState('dark');

  const toggleColorMode = () => {
    setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
  };

  const theme = useMemo(() => 
    createTheme({
      palette: {
        mode,
        primary: {
          main: mode === 'dark' ? '#4a6cf7' : '#2a4cdf',
          light: mode === 'dark' ? '#6a8cff' : '#4a6cf7',
          dark: mode === 'dark' ? '#3a5fff' : '#1a3acf',
          contrastText: mode === 'dark' ? '#ffffff' : '#ffffff'
        },
        secondary: {
          main: mode === 'dark' ? '#ff6b6b' : '#ff4757',
          light: mode === 'dark' ? '#ff7b7b' : '#ff6b6b',
          dark: mode === 'dark' ? '#ff5252' : '#ff3636',
          contrastText: mode === 'dark' ? '#ffffff' : '#ffffff'
        },
        background: {
          default: mode === 'dark' ? '#0a0a2a' : '#f4f6f9',
          paper: mode === 'dark' ? '#1a1a3a' : '#ffffff',
        },
        text: {
          primary: mode === 'dark' ? '#e0e0ff' : '#1a1a3a',
          secondary: mode === 'dark' ? 'rgba(224, 224, 255, 0.7)' : 'rgba(26, 26, 58, 0.6)',
          disabled: mode === 'dark' ? 'rgba(224, 224, 255, 0.4)' : 'rgba(26, 26, 58, 0.4)'
        },
        action: {
          active: mode === 'dark' ? '#4a6cf7' : '#2a4cdf',
          hover: mode === 'dark' ? 'rgba(74, 108, 247, 0.1)' : 'rgba(42, 76, 223, 0.05)',
          selected: mode === 'dark' ? 'rgba(74, 108, 247, 0.2)' : 'rgba(42, 76, 223, 0.1)',
          disabled: mode === 'dark' ? 'rgba(224, 224, 255, 0.3)' : 'rgba(26, 26, 58, 0.3)',
          disabledBackground: mode === 'dark' ? 'rgba(74, 108, 247, 0.1)' : 'rgba(42, 76, 223, 0.05)'
        },
      },
      typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        h1: {
          color: mode === 'dark' ? '#e0e0ff' : '#1a1a3a',
          fontWeight: 700,
        },
        h2: {
          color: mode === 'dark' ? '#e0e0ff' : '#1a1a3a',
          fontWeight: 600,
        },
        body1: {
          color: mode === 'dark' ? '#e0e0ff' : '#1a1a3a',
        },
        body2: {
          color: mode === 'dark' ? 'rgba(224, 224, 255, 0.7)' : 'rgba(26, 26, 58, 0.6)',
        }
      },
      components: {
        MuiCssBaseline: {
          styleOverrides: `
            body {
              background: ${
                mode === 'dark' 
                  ? 'linear-gradient(135deg, #0a0a2a 0%, #1a1a3a 100%)' 
                  : 'linear-gradient(135deg, #f4f6f9 0%, #e0e4f0 100%)'
              };
              background-attachment: fixed;
              min-height: 100vh;
              color: ${mode === 'dark' ? '#e0e0ff' : '#1a1a3a'};
              transition: background 0.3s ease, color 0.3s ease;
            }
          `,
        },
        MuiDrawer: {
          styleOverrides: {
            paper: {
              background: mode === 'dark' 
                ? 'rgba(26, 26, 58, 0.9)' 
                : 'rgba(240, 240, 255, 0.9)',
              backdropFilter: 'blur(10px)',
              borderRight: `1px solid ${mode === 'dark' ? 'rgba(74, 108, 247, 0.2)' : 'rgba(42, 76, 223, 0.1)'}`,
            },
          },
        },
        MuiListItem: {
          styleOverrides: {
            root: {
              '&:hover': {
                backgroundColor: mode === 'dark' 
                  ? 'rgba(74, 108, 247, 0.1)' 
                  : 'rgba(42, 76, 223, 0.05)',
                transition: 'background-color 0.3s ease',
              }
            }
          }
        },
        MuiIconButton: {
          styleOverrides: {
            root: {
              color: mode === 'dark' ? '#e0e0ff' : '#1a1a3a',
              transition: 'color 0.3s ease',
              '&:hover': {
                backgroundColor: mode === 'dark' 
                  ? 'rgba(74, 108, 247, 0.1)' 
                  : 'rgba(42, 76, 223, 0.05)',
                transition: 'background-color 0.3s ease',
              }
            }
          }
        }
      },
    }), 
    [mode]
  );

  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }
    setDrawerOpen(open);
  };

  const sidebarItems = [
    { text: 'Home', icon: <DashboardIcon />, path: '/' },
    { text: 'Trading Dashboard', icon: <AutoGraphIcon />, path: '/dashboard' },
    { text: 'Market Analysis', icon: <AnalyticsIcon />, path: '/analysis' },
    { text: 'LLM Analysis', icon: <BrainIcon />, path: '/llm-analysis' },
    { text: 'Add New Model', icon: <AddCircleOutlineIcon />, path: '/new-model' },
    { text: 'AutoML Analysis', icon: <AnalyticsIcon />, path: '/automl-analysis' },
    { text: 'Model Results', icon: <AnalyticsIcon />, path: '/model-results' },
    { text: 'AI Dashboard', icon: <DashboardIcon />, path: '/ai-dashboard' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' }
  ];

  const drawerContent = (
    <Box
      sx={{ width: 250 }}
      role="presentation"
      onClick={toggleDrawer(false)}
      onKeyDown={toggleDrawer(false)}
    >
      <List>
        {sidebarItems.map((item, index) => (
          <motion.div
            key={item.text}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <ListItem 
              button 
              component={Link} 
              to={item.path}
              sx={{
                '&:hover': {
                  backgroundColor: mode === 'dark' 
                    ? 'rgba(74, 108, 247, 0.1)' 
                    : 'rgba(42, 76, 223, 0.05)',
                },
              }}
            >
              <ListItemIcon sx={{ color: theme.palette.primary.main }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text} 
                primaryTypographyProps={{ 
                  color: 'text.primary',
                  fontWeight: 'medium'
                }} 
              />
            </ListItem>
          </motion.div>
        ))}
      </List>
      
      {/* Theme Toggle */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        p: 2 
      }}>
        <FormControlLabel
          control={
            <Switch
              checked={mode === 'dark'}
              onChange={toggleColorMode}
              color="primary"
            />
          }
          label={
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              color: 'text.primary' 
            }}>
              {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
              <Typography sx={{ ml: 1 }}>
                {mode === 'dark' ? 'Light Mode' : 'Dark Mode'}
              </Typography>
            </Box>
          }
        />
      </Box>
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        minHeight: '100vh',
        background: theme.palette.background.default
      }}>
        <AppBar 
          position="static" 
          sx={{ 
            background: mode === 'dark' 
              ? 'rgba(58, 95, 255, 0.1)' 
              : 'rgba(42, 76, 223, 0.05)', 
            backdropFilter: 'blur(10px)',
            borderBottom: `1px solid ${
              mode === 'dark' 
                ? 'rgba(74, 108, 247, 0.2)' 
                : 'rgba(42, 76, 223, 0.1)'
            }`
          }}
        >
          <Toolbar>
            <IconButton
              size="large"
              edge="start"
              color="inherit"
              aria-label="menu"
              sx={{ mr: 2 }}
              onClick={toggleDrawer(true)}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: theme.palette.primary.main }}>
              ZyraTech AI Trading Platform
            </Typography>
            <IconButton 
              color="inherit" 
              onClick={toggleColorMode}
              sx={{ ml: 1 }}
            >
              {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
          </Toolbar>
        </AppBar>

        <Drawer
          anchor="left"
          open={drawerOpen}
          onClose={toggleDrawer(false)}
        >
          {drawerContent}
        </Drawer>

        <Container 
          component="main" 
          maxWidth={false} 
          disableGutters
          sx={{ 
            flexGrow: 1, 
            py: 3,
            background: 'transparent',
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          <Box sx={{ flexGrow: 1 }}>
            <Routes>
              <Route path="/dashboard" element={
                <ProtectedRoute>
                  <TradingDashboard />
                </ProtectedRoute>
              } />
              <Route path="/analysis" element={
                <ProtectedRoute>
                  <MarketAnalyzer />
                </ProtectedRoute>
              } />
              <Route path="/llm-analysis" element={
                <ProtectedRoute>
                  <LLMModelAnalysis />
                </ProtectedRoute>
              } />
              <Route path="/new-model" element={
                <ProtectedRoute>
                  <LLMNewModel />
                </ProtectedRoute>
              } />
              <Route path="/automl-analysis" element={
                <ProtectedRoute>
                  <AutoMLAnalysis />
                </ProtectedRoute>
              } />
              <Route path="/model-results" element={
                <ProtectedRoute>
                  <ModelResults />
                </ProtectedRoute>
              } />
              <Route path="/ai-dashboard" element={
                <ProtectedRoute>
                  <AIDashboard />
                </ProtectedRoute>
              } />
              <Route path="/settings" element={
                <ProtectedRoute>
                  <Settings />
                </ProtectedRoute>
              } />
              <Route path="/" element={
                <ProtectedRoute>
                  <Navigate to="/dashboard" replace />
                </ProtectedRoute>
              } />
            </Routes>
          </Box>
        </Container>

        <Box 
          component="footer" 
          sx={{ 
            py: 2, 
            px: 2, 
            mt: 'auto', 
            background: mode === 'dark' 
              ? 'rgba(26, 26, 58, 0.5)' 
              : 'rgba(42, 76, 223, 0.05)',
            borderTop: `1px solid ${
              mode === 'dark' 
                ? 'rgba(74, 108, 247, 0.2)' 
                : 'rgba(42, 76, 223, 0.1)'
            }`
          }}
        >
          <Typography variant="body2" color="text.secondary" align="center">
            2024 Market Analysis Platform. All rights reserved By ZyraTech.
          </Typography>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
