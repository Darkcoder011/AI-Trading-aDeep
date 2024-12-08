import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  IconButton, 
  InputAdornment,
  Paper
} from '@mui/material';
import { motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import SecurityIcon from '@mui/icons-material/Security';
import { alpha } from '@mui/material/styles';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const { login } = useAuth();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (username === 'admin' && password === 'LLM@Market2024') {
      login();
    } else {
      setError('Invalid credentials 🚫');
    }
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        when: "beforeChildren",
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: { duration: 0.3 }
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: (theme) => `linear-gradient(45deg, 
          ${alpha(theme.palette.primary.main, 0.1)} 0%, 
          ${alpha(theme.palette.secondary.main, 0.1)} 100%)`
      }}
    >
      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
      >
        <Paper
          elevation={24}
          sx={{
            p: 4,
            width: '100%',
            maxWidth: 400,
            borderRadius: 4,
            backdropFilter: 'blur(10px)',
            background: (theme) => alpha(theme.palette.background.paper, 0.8),
            border: '1px solid',
            borderColor: 'divider'
          }}
        >
          <motion.div variants={itemVariants}>
            <Box sx={{ mb: 3, textAlign: 'center' }}>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ 
                  type: "spring",
                  stiffness: 260,
                  damping: 20 
                }}
              >
                <RocketLaunchIcon 
                  sx={{ 
                    fontSize: 50,
                    mb: 2,
                    color: 'primary.main'
                  }} 
                />
              </motion.div>
              <Typography variant="h4" component="h1" gutterBottom>
                Welcome Back! 👋
              </Typography>
              <Typography variant="subtitle1" color="text.secondary">
                Enter your credentials to access the platform 🚀
              </Typography>
            </Box>
          </motion.div>

          <form onSubmit={handleSubmit}>
            <motion.div variants={itemVariants}>
              <TextField
                fullWidth
                label="Username"
                variant="outlined"
                margin="normal"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SecurityIcon color="primary" />
                    </InputAdornment>
                  ),
                }}
              />
            </motion.div>

            <motion.div variants={itemVariants}>
              <TextField
                fullWidth
                label="Password"
                type={showPassword ? 'text' : 'password'}
                variant="outlined"
                margin="normal"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <LockOutlinedIcon color="primary" />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowPassword(!showPassword)}
                        edge="end"
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
              />
            </motion.div>

            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                <Typography color="error" sx={{ mt: 1, textAlign: 'center' }}>
                  {error}
                </Typography>
              </motion.div>
            )}

            <motion.div variants={itemVariants}>
              <Button
                fullWidth
                variant="contained"
                size="large"
                type="submit"
                sx={{ 
                  mt: 3, 
                  mb: 2,
                  background: (theme) => `linear-gradient(45deg, 
                    ${theme.palette.primary.main} 30%, 
                    ${theme.palette.secondary.main} 90%)`,
                  '&:hover': {
                    background: (theme) => `linear-gradient(45deg, 
                      ${theme.palette.primary.dark} 30%, 
                      ${theme.palette.secondary.dark} 90%)`,
                  }
                }}
              >
                Login 🔐
              </Button>
            </motion.div>
          </form>
        </Paper>
      </motion.div>
    </Box>
  );
};

export default Login;
