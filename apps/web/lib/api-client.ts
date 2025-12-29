import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      window.location.href = '/auth/login';
    }
    return Promise.reject(error);
  }
);

export const evolutionApi = {
  getStatus: () => apiClient.get('/api/v1/evolution/status'),
  getMetrics: (limit = 100) => apiClient.get(`/api/v1/evolution/metrics?limit=${limit}`),
  getPopulation: () => apiClient.get('/api/v1/evolution/population'),
  startTraining: (config: any) => apiClient.post('/api/v1/evolution/start', config),
  stopTraining: () => apiClient.post('/api/v1/evolution/stop'),
};

export const redTeamApi = {
  getAttacks: (limit = 50) => apiClient.get(`/api/v1/red-team/attacks?limit=${limit}`),
  getStrategies: () => apiClient.get('/api/v1/red-team/strategies'),
  getMetrics: () => apiClient.get('/api/v1/red-team/metrics'),
};

export const blueTeamApi = {
  getDetections: () => apiClient.get('/api/v1/blue-team/detections'),
  getRules: () => apiClient.get('/api/v1/blue-team/rules'),
  getMetrics: () => apiClient.get('/api/v1/blue-team/metrics'),
};

export const cyberRangeApi = {
  getStatus: () => apiClient.get('/api/v1/cyber-range/status'),
  reset: () => apiClient.post('/api/v1/cyber-range/reset'),
};

export const analyticsApi = {
  getDashboard: () => apiClient.get('/api/v1/analytics/dashboard'),
  getTrends: (timeframe = '24h') => apiClient.get(`/api/v1/analytics/trends?timeframe=${timeframe}`),
};
