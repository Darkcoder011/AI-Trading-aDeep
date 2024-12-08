import React, { createContext, useContext, useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const AuthContext = createContext(null);

// Protected routes list
const protectedRoutes = [
  '/dashboard',
  '/analysis',
  '/llm-analysis',
  '/new-model',
  '/automl-analysis',
  '/model-results',
  '/ai-dashboard',
  '/settings',
  '/home'
];

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isSecurityVerified, setIsSecurityVerified] = useState(false);
  const [securityPassword, setSecurityPassword] = useState('Market@2024');
  const navigate = useNavigate();
  const location = useLocation();

  // Check authentication on route change
  useEffect(() => {
    const isProtectedRoute = protectedRoutes.some(route => 
      location.pathname.startsWith(route)
    );

    if (isProtectedRoute) {
      if (!isAuthenticated) {
        navigate('/', { replace: true });
      } else if (!isSecurityVerified) {
        navigate('/enter-password', { replace: true });
      }
    }
  }, [location, isAuthenticated, isSecurityVerified, navigate]);

  const login = () => {
    setIsAuthenticated(true);
    navigate('/enter-password', { replace: true });
  };

  const verifySecurityPassword = (password) => {
    if (password === securityPassword) {
      setIsSecurityVerified(true);
      navigate('/dashboard', { replace: true });
      return true;
    }
    return false;
  };

  const changeSecurityPassword = (currentPassword, newPassword) => {
    if (currentPassword === securityPassword) {
      setSecurityPassword(newPassword);
      return true;
    }
    return false;
  };

  const logout = () => {
    setIsAuthenticated(false);
    setIsSecurityVerified(false);
    navigate('/', { replace: true });
  };

  return (
    <AuthContext.Provider 
      value={{ 
        isAuthenticated, 
        isSecurityVerified, 
        login, 
        verifySecurityPassword, 
        changeSecurityPassword,
        logout 
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
