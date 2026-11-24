import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor: attach token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: handle 401 unauthorized
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Optional auto-logout or redirect
    }
    return Promise.reject(error);
  }
);

// Authentication
export const authService = {
  login: (email, password) => api.post('/auth/login', { email, password }),
  register: (userData) => api.post('/auth/register', userData),
  getMe: () => api.get('/auth/me'),
  forgotPassword: (email) => api.post('/auth/forgot-password', { email }),
};

// Patients
export const patientService = {
  getPatients: (params) => api.get('/patients', { params }),
  getPatientById: (id) => api.get(`/patients/${id}`),
  createPatient: (patientData) => api.post('/patients', patientData),
  updatePatient: (id, data) => api.put(`/patients/${id}`, data),
  deletePatient: (id) => api.delete(`/patients/${id}`),
};

// Scans & Predictions
export const scanService = {
  uploadScan: (formData, onProgress) =>
    api.post('/scans/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: onProgress,
    }),
  predictDemo: (demoId, patientInfo) =>
    api.post(`/scans/predict-demo/${demoId}`, patientInfo),
  getPredictions: (params) => api.get('/predictions', { params }),
  getPredictionById: (id) => api.get(`/predictions/${id}`),
  updateDoctorNotes: (id, data) => api.put(`/predictions/${id}/notes`, data),
  deletePrediction: (id) => api.delete(`/predictions/${id}`),
};

// Reports
export const reportService = {
  generateReport: (predictionId) => api.post(`/reports/generate/${predictionId}`),
  getReportDownloadUrl: (fileName) => `${API_BASE}/reports/${fileName}/download`,
};

// Analytics & Metrics
export const analyticsService = {
  getDashboardStats: () => api.get('/analytics/dashboard'),
  getModelMetrics: () => api.get('/analytics/model-metrics'),
};

// Audit Logs
export const auditService = {
  getAuditLogs: (params) => api.get('/audit-logs', { params }),
};

export default api;
