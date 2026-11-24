import React, { createContext, useContext, useState, useEffect } from 'react';
import { authService } from '../services/api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [loading, setLoading] = useState(true);

  // Restore authenticated session
  useEffect(() => {
    const fetchUser = async () => {
      if (token) {
        try {
          const res = await authService.getMe();
          setUser(res.data.user);
        } catch (err) {
          console.warn('Session expired or invalid token:', err.message);
          logout();
        }
      } else {
        // Auto-seed default guest doctor if none logged in for instant preview
        setUser({
          id: 'user-001',
          name: 'Dr. Sarah Mitchell, MD',
          email: 'doctor@hospital.org',
          role: 'doctor',
          hospital: 'Memorial Sloan Cancer Center',
          department: 'Diagnostic Radiology & Oncology',
        });
      }
      setLoading(false);
    };

    fetchUser();
  }, [token]);

  const login = async (email, password) => {
    const res = await authService.login(email, password);
    const { token: receivedToken, user: receivedUser } = res.data;
    localStorage.setItem('token', receivedToken);
    setToken(receivedToken);
    setUser(receivedUser);
    return receivedUser;
  };

  const register = async (userData) => {
    const res = await authService.register(userData);
    const { token: receivedToken, user: receivedUser } = res.data;
    localStorage.setItem('token', receivedToken);
    setToken(receivedToken);
    setUser(receivedUser);
    return receivedUser;
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
  };

  // Fast demo role switcher (Admin / Doctor / Researcher)
  const switchDemoRole = (role) => {
    const roleProfiles = {
      doctor: {
        id: 'user-001',
        name: 'Dr. Sarah Mitchell, MD',
        email: 'doctor@hospital.org',
        role: 'doctor',
        hospital: 'Memorial Sloan Cancer Center',
        department: 'Diagnostic Radiology & Oncology',
      },
      researcher: {
        id: 'user-002',
        name: 'Prof. David Vance, PhD',
        email: 'researcher@oncology.edu',
        role: 'researcher',
        hospital: 'National Cancer Research Institute',
        department: 'Computational Oncology & AI Labs',
      },
      admin: {
        id: 'user-003',
        name: 'Elena Rostova (Admin)',
        email: 'admin@cancercenter.org',
        role: 'admin',
        hospital: 'Global Health Imaging Consortium',
        department: 'Medical Systems Administration',
      },
    };

    const newProfile = roleProfiles[role] || roleProfiles.doctor;
    setUser(newProfile);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        isAuthenticated: !!user,
        role: user?.role || 'doctor',
        login,
        register,
        logout,
        switchDemoRole,
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
