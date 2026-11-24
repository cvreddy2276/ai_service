const express = require('express');
const router = express.Router();
const { getDashboardStats, getModelMetrics } = require('../controllers/analyticsController');
const { protect } = require('../middleware/auth');

router.use(protect);

router.get('/dashboard', getDashboardStats);
router.get('/model-metrics', getModelMetrics);

module.exports = router;
