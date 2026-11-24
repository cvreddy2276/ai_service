const express = require('express');
const router = express.Router();
const {
  uploadAndPredict,
  predictDemo,
  getPredictions,
  getPredictionById,
  updateNotes,
  deletePrediction,
} = require('../controllers/predictionController');
const { protect, authorize } = require('../middleware/auth');
const upload = require('../middleware/upload');

// Public or Protected routes
router.use(protect);

router.post('/upload', upload.single('file'), uploadAndPredict);
router.post('/predict-demo/:demoId', predictDemo);
router.get('/', getPredictions);
router.get('/:id', getPredictionById);
router.put('/:id/notes', authorize('doctor', 'admin'), updateNotes);
router.delete('/:id', authorize('admin'), deletePrediction);

module.exports = router;
