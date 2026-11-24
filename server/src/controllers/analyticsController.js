const Prediction = require('../models/Prediction');
const Patient = require('../models/Patient');
const { store } = require('../services/dataStore');
const { isUsingMemoryStore } = require('../config/db');
const aiService = require('../services/aiService');

// @desc    Get dashboard metrics & clinical statistics
// @route   GET /api/analytics/dashboard
// @access  Private
exports.getDashboardStats = async (req, res) => {
  try {
    let predictions = [];
    let patients = [];

    if (isUsingMemoryStore()) {
      predictions = store.predictions.filter((p) => ['pCR likely', 'pCR unlikely'].includes(p.prediction));
      patients = store.patients;
    } else {
      predictions = await Prediction.find({ prediction: { $in: ['pCR likely', 'pCR unlikely'] } });
      patients = await Patient.find();
      if (predictions.length === 0) predictions = store.predictions.filter((p) => ['pCR likely', 'pCR unlikely'].includes(p.prediction));
      if (patients.length === 0) patients = store.patients;
    }

    const totalScans = predictions.length;
    const pcrLikelyCases = predictions.filter((p) => p.isPcrLikely || p.prediction === 'pCR likely').length;
    const pcrUnlikelyCases = totalScans - pcrLikelyCases;

    const avgLatency = totalScans > 0
      ? Math.round(predictions.reduce((acc, p) => acc + (p.processingTimeMs || 0), 0) / totalScans)
      : 0;

    // Monthly volume breakdown
    const monthlyCounts = new Map();
    for (const prediction of predictions) {
      const date = prediction.createdAt ? new Date(prediction.createdAt) : null;
      if (!date || Number.isNaN(date.getTime())) continue;
      const month = date.toLocaleString('en', { month: 'short', year: 'numeric' });
      const current = monthlyCounts.get(month) || { month, total: 0, pcrLikely: 0, pcrUnlikely: 0, timestamp: date.getTime() };
      current.total += 1;
      current[prediction.isPcrLikely || prediction.prediction === 'pCR likely' ? 'pcrLikely' : 'pcrUnlikely'] += 1;
      monthlyCounts.set(month, current);
    }
    const monthlyTrends = [...monthlyCounts.values()]
      .sort((a, b) => a.timestamp - b.timestamp)
      .slice(-6)
      .map(({ timestamp, ...entry }) => entry);

    res.status(200).json({
      success: true,
      data: {
        totalScans,
        totalPatients: patients.length,
        pcrLikelyCases,
        pcrUnlikelyCases,
        avgLatencyMs: avgLatency,
        monthlyTrends,
        recentPredictions: predictions.slice(0, 6),
      },
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Get verified model evaluation metrics
// @route   GET /api/analytics/model-metrics
// @access  Private
exports.getModelMetrics = async (req, res) => {
  try {
    const metrics = await aiService.getModelMetrics();
    return res.status(200).json({ success: true, data: { metrics } });
  } catch (err) {
    return res.status(err.response?.status || 503).json({
      success: false,
      message: err.response?.data?.detail || 'Verified held-out model metrics are unavailable.',
    });
  }
};
