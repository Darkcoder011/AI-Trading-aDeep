import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  IconButton,
  InputAdornment,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import VpnKeyIcon from '@mui/icons-material/VpnKey';
import ShieldIcon from '@mui/icons-material/Shield';
import AdminPanelSettingsIcon from '@mui/icons-material/AdminPanelSettings';
import { alpha } from '@mui/material/styles';

const EnterPassword = () => {
  const [securityPassword, setSecurityPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const { verifySecurityPassword, changeSecurityPassword } = useAuth();

  // Password change states
  const [isChangePasswordOpen, setIsChangePasswordOpen] = useState(false);
  const [adminPassword, setAdminPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showAdminPassword, setShowAdminPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [changePasswordError, setChangePasswordError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (verifySecurityPassword(securityPassword)) {
      setError('');
    } else {
      setError('Invalid security password ');
    }
  };

  const handleChangePasswordClick = () => {
    setIsChangePasswordOpen(true);
    setAdminPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setChangePasswordError('');
  };

  const handleChangePasswordSubmit = () => {
    if (!adminPassword || !newPassword || !confirmPassword) {
      setChangePasswordError('Please fill in all fields ');
      return;
    }

    if (newPassword !== confirmPassword) {
      setChangePasswordError('New passwords do not match ');
      return;
    }

    if (newPassword.length < 8) {
      setChangePasswordError('Password must be at least 8 characters long ');
      return;
    }

    if (changeSecurityPassword(adminPassword, newPassword)) {
      setIsChangePasswordOpen(false);
      setError('Password successfully changed! ');
    } else {
      setChangePasswordError('Invalid admin password ');
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

  const iconAnimation = {
    hidden: { rotate: -180, scale: 0 },
    visible: {
      rotate: 0,
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 260,
        damping: 20
      }
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
          ${alpha(theme.palette.secondary.main, 0.1)} 0%, 
          ${alpha(theme.palette.primary.main, 0.1)} 100%)`
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
              <motion.div variants={iconAnimation}>
                <ShieldIcon 
                  sx={{ 
                    fontSize: 50,
                    mb: 2,
                    color: 'secondary.main'
                  }} 
                />
              </motion.div>
              <Typography variant="h4" component="h1" gutterBottom>
                Security Check 
              </Typography>
              <Typography variant="subtitle1" color="text.secondary">
                Enter your security password to proceed 
              </Typography>
            </Box>
          </motion.div>

          <form onSubmit={handleSubmit}>
            <motion.div variants={itemVariants}>
              <TextField
                fullWidth
                label="Security Password"
                type={showPassword ? 'text' : 'password'}
                variant="outlined"
                margin="normal"
                value={securityPassword}
                onChange={(e) => setSecurityPassword(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <VpnKeyIcon color="secondary" />
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

            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <Typography color="error" sx={{ mt: 1, textAlign: 'center' }}>
                    {error}
                  </Typography>
                </motion.div>
              )}
            </AnimatePresence>

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
                    ${theme.palette.secondary.main} 30%, 
                    ${theme.palette.primary.main} 90%)`,
                  '&:hover': {
                    background: (theme) => `linear-gradient(45deg, 
                      ${theme.palette.secondary.dark} 30%, 
                      ${theme.palette.primary.dark} 90%)`,
                  }
                }}
              >
                Verify Security 
              </Button>
            </motion.div>
          </form>

          <motion.div 
            variants={itemVariants}
            style={{ textAlign: 'center', marginTop: '16px' }}
          >
            <Typography variant="body2" color="text.secondary">
              Protected by advanced security 
            </Typography>
            <Button
              color="primary"
              onClick={handleChangePasswordClick}
              sx={{ mt: 2 }}
              startIcon={<AdminPanelSettingsIcon />}
            >
              Admin: Change Password 
            </Button>
          </motion.div>
        </Paper>
      </motion.div>

      {/* Change Password Dialog */}
      <Dialog 
        open={isChangePasswordOpen} 
        onClose={() => setIsChangePasswordOpen(false)}
        PaperProps={{
          sx: {
            borderRadius: 2,
            maxWidth: 400,
            width: '100%'
          }
        }}
      >
        <DialogTitle>
          <Box sx={{ textAlign: 'center' }}>
            <AdminPanelSettingsIcon 
              sx={{ 
                fontSize: 40, 
                color: 'primary.main',
                mb: 1
              }} 
            />
            <Typography variant="h6">
              Admin: Change Security Password 
            </Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Admin Password"
            type={showAdminPassword ? 'text' : 'password'}
            margin="normal"
            value={adminPassword}
            onChange={(e) => setAdminPassword(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <AdminPanelSettingsIcon color="primary" />
                </InputAdornment>
              ),
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={() => setShowAdminPassword(!showAdminPassword)}
                    edge="end"
                  >
                    {showAdminPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          <TextField
            fullWidth
            label="New Security Password"
            type={showNewPassword ? 'text' : 'password'}
            margin="normal"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <VpnKeyIcon color="secondary" />
                </InputAdornment>
              ),
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    edge="end"
                  >
                    {showNewPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          <TextField
            fullWidth
            label="Confirm New Password"
            type={showConfirmPassword ? 'text' : 'password'}
            margin="normal"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <VpnKeyIcon color="secondary" />
                </InputAdornment>
              ),
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    edge="end"
                  >
                    {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          {changePasswordError && (
            <Typography color="error" align="center" sx={{ mt: 2 }}>
              {changePasswordError}
            </Typography>
          )}
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 3 }}>
          <Button 
            onClick={() => setIsChangePasswordOpen(false)}
            variant="outlined"
            startIcon={<LockOutlinedIcon />}
          >
            Cancel 
          </Button>
          <Button 
            onClick={handleChangePasswordSubmit}
            variant="contained"
            startIcon={<AdminPanelSettingsIcon />}
            sx={{
              background: (theme) => `linear-gradient(45deg, 
                ${theme.palette.secondary.main} 30%, 
                ${theme.palette.primary.main} 90%)`,
              '&:hover': {
                background: (theme) => `linear-gradient(45deg, 
                  ${theme.palette.secondary.dark} 30%, 
                  ${theme.palette.primary.dark} 90%)`,
              }
            }}
          >
            Change Password 
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default EnterPassword;
