const express = require('express');
const router = express.Router();
const { generateReport, downloadReport } = require('../controllers/reportController');
const { protect } = require('../middleware/auth');

router.post('/generate/:predictionId', protect, generateReport);
router.get('/:fileName/download', downloadReport);

module.exports = router;
