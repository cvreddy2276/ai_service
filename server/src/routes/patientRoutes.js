const express = require('express');
const router = express.Router();
const {
  getPatients,
  getPatientById,
  createPatient,
  updatePatient,
  deletePatient,
} = require('../controllers/patientController');
const { protect, authorize } = require('../middleware/auth');

router.use(protect);

router.route('/').get(getPatients).post(authorize('doctor', 'admin'), createPatient);

router
  .route('/:id')
  .get(getPatientById)
  .put(authorize('doctor', 'admin'), updatePatient)
  .delete(authorize('admin'), deletePatient);

module.exports = router;
