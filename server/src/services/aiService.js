const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');
const path = require('path');

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

class AIServiceClient {
  constructor() {
    this.client = axios.create({
      baseURL: AI_SERVICE_URL,
      timeout: 60000, // 60s timeout for large 3D volumetric tensor computation
    });
  }

  async checkHealth() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (err) {
      return { status: 'offline', error: err.message };
    }
  }

  async getModelMetrics() {
    const response = await this.client.get('/metrics');
    return response.data;
  }

  async predictNiftiFile(filePath, originalFilename) {
    try {
      const formData = new FormData();
      formData.append('file', fs.createReadStream(filePath), {
        filename: originalFilename,
        contentType: 'application/octet-stream',
      });

      const response = await this.client.post('/predict', formData, {
        headers: {
          ...formData.getHeaders(),
        },
      });

      return response.data;
    } catch (err) {
      throw new Error(`AI inference unavailable: ${err.response?.data?.detail || err.message}`);
    }
  }

  async predictDemo(demoId) {
    try {
      const response = await this.client.post(`/predict-demo/${demoId}`);
      return response.data;
    } catch (err) {
      throw new Error(`Demo inference unavailable: ${err.response?.data?.detail || err.message}`);
    }
  }
}

module.exports = new AIServiceClient();
