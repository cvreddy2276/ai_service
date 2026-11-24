const { v4: uuidv4 } = require('uuid');
const Prediction = require('../models/Prediction');
const Patient = require('../models/Patient');
const { store } = require('../services/dataStore');
const { isUsingMemoryStore } = require('../config/db');
const aiService = require('../services/aiService');
const { logAuditEvent } = require('../middleware/audit');

// @desc    Upload NIfTI scan and execute AI prediction
// @route   POST /api/scans/upload
// @access  Private
exports.uploadAndPredict = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'Please upload a NIfTI scan file (.nii or .nii.gz).' });
    }

    const { patientId = 'PT-78204', patientName = 'Margaret Chen' } = req.body;
    const originalFilename = req.file.originalname;
    const filePath = req.file.path;

    console.log(`[Prediction] Received file upload: ${originalFilename} (${req.file.size} bytes) for patient ${patientId}`);

    // Call AI Service
    const aiResult = await aiService.predictNiftiFile(filePath, originalFilename);

    const predictionPayload = {
      patientId,
      patientName,
      scanFileName: originalFilename,
      scanFilePath: filePath,
      prediction: aiResult.prediction,
      isPcrLikely: aiResult.is_pcr_likely,
      confidence: aiResult.confidence,
      confidencePercentage: aiResult.confidence_percentage,
      riskLevel: aiResult.risk_level,
      pcrProbability: aiResult.pcr_probability,
      tumorVolumeCm3: aiResult.tumor_volume_cm3,
      peakSliceIndex: aiResult.peak_slice_index,
      totalSlices: aiResult.total_slices,
      dimensions: aiResult.dimensions,
      voxelSpacing: aiResult.voxel_spacing,
      modelVersion: aiResult.model_version || 'pcr-research-v0.1',
      processingTimeMs: aiResult.processing_time_ms,
      findings: aiResult.findings,
      recommendations: aiResult.recommendations,
      keySlices: (aiResult.key_slices || []).map((s) => ({
        sliceIndex: s.slice_index,
        plane: s.plane,
        imageBase64: s.image_base64,
        gradcamBase64: s.gradcam_base64,
        isPeakAttribution: s.is_peak_attribution,
      })),
      doctorNotes: '',
      status: 'Pending Review',
      createdAt: new Date().toISOString(),
    };

    if (isUsingMemoryStore()) {
      const newPrediction = { _id: `pred-${uuidv4().substring(0, 8)}`, ...predictionPayload };
      store.predictions.unshift(newPrediction);

      // Update patient stats in store
      const patient = store.patients.find((p) => p.patientId === patientId);
      if (patient) {
        patient.scanCount = (patient.scanCount || 0) + 1;
        patient.latestRiskLevel = 'Unclassified';
      }

      await logAuditEvent({
        action: 'SCAN_UPLOAD',
        user: req.user,
        req,
        resourceType: 'Scan',
        resourceId: newPrediction._id,
        details: `Uploaded ${originalFilename} & inferred: ${aiResult.prediction} (${aiResult.confidence_percentage}%)`,
      });

      return res.status(201).json({ success: true, data: newPrediction });
    }

    const prediction = await Prediction.create(predictionPayload);

    // Update patient
    await Patient.findOneAndUpdate(
      { patientId },
      {
        $inc: { scanCount: 1 },
        latestRiskLevel: 'Unclassified',
      }
    );

    await logAuditEvent({
      action: 'SCAN_UPLOAD',
      user: req.user,
      req,
      resourceType: 'Scan',
      resourceId: prediction._id,
      details: `Uploaded ${originalFilename} & inferred: ${aiResult.prediction} (${aiResult.confidence_percentage}%)`,
    });

    res.status(201).json({ success: true, data: prediction });
  } catch (err) {
    console.error('[PredictionController] Error during scan upload/predict:', err);
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Execute instant prediction on demo scan
// @route   POST /api/scans/predict-demo/:demoId
// @access  Private
exports.predictDemo = async (req, res) => {
  try {
    const { demoId } = req.params;
    const { patientId = 'PT-89421', patientName = 'Eleanor Vance' } = req.body;

    console.log(`[Prediction] Running demo scan inference: ${demoId} for patient ${patientId}`);
    const aiResult = await aiService.predictDemo(demoId);

    const predictionPayload = {
      patientId,
      patientName,
      scanFileName: `Demo_${demoId}_DCE_Subtraction.nii.gz`,
      scanFilePath: `data/demo/${demoId}.nii.gz`,
      prediction: aiResult.prediction,
      isPcrLikely: aiResult.is_pcr_likely,
      confidence: aiResult.confidence,
      confidencePercentage: aiResult.confidence_percentage,
      riskLevel: aiResult.risk_level,
      pcrProbability: aiResult.pcr_probability,
      tumorVolumeCm3: aiResult.tumor_volume_cm3,
      peakSliceIndex: aiResult.peak_slice_index,
      totalSlices: aiResult.total_slices,
      dimensions: aiResult.dimensions,
      voxelSpacing: aiResult.voxel_spacing,
      modelVersion: aiResult.model_version || 'pcr-research-v0.1',
      processingTimeMs: aiResult.processing_time_ms,
      findings: aiResult.findings,
      recommendations: aiResult.recommendations,
      keySlices: (aiResult.key_slices || []).map((s) => ({
        sliceIndex: s.slice_index,
        plane: s.plane,
        imageBase64: s.image_base64,
        gradcamBase64: s.gradcam_base64,
        isPeakAttribution: s.is_peak_attribution,
      })),
      doctorNotes: '',
      status: 'Pending Review',
      createdAt: new Date().toISOString(),
    };

    if (isUsingMemoryStore()) {
      const newPrediction = { _id: `pred-${uuidv4().substring(0, 8)}`, ...predictionPayload };
      store.predictions.unshift(newPrediction);

      const patient = store.patients.find((p) => p.patientId === patientId);
      if (patient) {
        patient.scanCount = (patient.scanCount || 0) + 1;
        patient.latestRiskLevel = 'Unclassified';
      }

      await logAuditEvent({
        action: 'PREDICTION_RUN',
        user: req.user,
        req,
        resourceType: 'Scan',
        resourceId: newPrediction._id,
        details: `Ran demo scan ${demoId} prediction: ${aiResult.prediction}`,
      });

      return res.status(201).json({ success: true, data: newPrediction });
    }

    const prediction = await Prediction.create(predictionPayload);

    await Patient.findOneAndUpdate(
      { patientId },
      {
        $inc: { scanCount: 1 },
        latestRiskLevel: 'Unclassified',
      }
    );

    await logAuditEvent({
      action: 'PREDICTION_RUN',
      user: req.user,
      req,
      resourceType: 'Scan',
      resourceId: prediction._id,
      details: `Ran demo scan ${demoId} prediction: ${aiResult.prediction}`,
    });

    res.status(201).json({ success: true, data: prediction });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Get all prediction results
// @route   GET /api/predictions
// @access  Private
exports.getPredictions = async (req, res) => {
  try {
    const { patientId, risk, verdict } = req.query;

    if (isUsingMemoryStore()) {
      let results = store.predictions.filter((p) => ['pCR likely', 'pCR unlikely'].includes(p.prediction));
      if (patientId) results = results.filter((p) => p.patientId === patientId);
      if (risk && risk !== 'all') results = results.filter((p) => (p.riskLevel || '').toLowerCase() === risk.toLowerCase());
      if (verdict && verdict !== 'all') results = results.filter((p) => p.prediction.toLowerCase() === verdict.toLowerCase());

      return res.status(200).json({ success: true, count: results.length, data: results });
    }

    const filter = { prediction: { $in: ['pCR likely', 'pCR unlikely'] } };
    if (patientId) filter.patientId = patientId;
    if (risk && risk !== 'all') filter.riskLevel = risk;
    if (verdict && verdict !== 'all') filter.prediction = verdict;

    const predictions = await Prediction.find(filter).sort({ createdAt: -1 });
    res.status(200).json({ success: true, count: predictions.length, data: predictions });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Get single prediction by ID
// @route   GET /api/predictions/:id
// @access  Private
exports.getPredictionById = async (req, res) => {
  try {
    const { id } = req.params;

    if (isUsingMemoryStore()) {
      const pred = store.predictions.find((p) => p._id === id);
      if (!pred) {
        return res.status(404).json({ success: false, message: 'Prediction record not found.' });
      }
      return res.status(200).json({ success: true, data: pred });
    }

    const prediction = await Prediction.findOne({ _id: id, prediction: { $in: ['pCR likely', 'pCR unlikely'] } });
    if (!prediction) {
      const memPred = store.predictions.find((p) => p._id === id);
      if (memPred) return res.status(200).json({ success: true, data: memPred });
      return res.status(404).json({ success: false, message: 'Prediction record not found.' });
    }

    res.status(200).json({ success: true, data: prediction });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Update doctor clinical review & notes
// @route   PUT /api/predictions/:id/notes
// @access  Private (Doctor / Admin)
exports.updateNotes = async (req, res) => {
  try {
    const { id } = req.params;
    const { doctorNotes, status = 'Confirmed' } = req.body;

    if (isUsingMemoryStore()) {
      const index = store.predictions.findIndex((p) => p._id === id);
      if (index === -1) {
        return res.status(404).json({ success: false, message: 'Prediction record not found.' });
      }

      store.predictions[index].doctorNotes = doctorNotes;
      store.predictions[index].status = status;
      store.predictions[index].reviewerName = req.user?.name || 'Attending Physician';

      await logAuditEvent({
        action: 'PREDICTION_REVIEW',
        user: req.user,
        req,
        resourceType: 'Prediction',
        resourceId: id,
        details: `Doctor updated notes and marked case as ${status}`,
      });

      return res.status(200).json({ success: true, data: store.predictions[index] });
    }

    const prediction = await Prediction.findByIdAndUpdate(
      id,
      {
        doctorNotes,
        status,
        reviewerName: req.user?.name || 'Attending Physician',
      },
      { new: true }
    );

    if (!prediction) {
      return res.status(404).json({ success: false, message: 'Prediction record not found.' });
    }

    await logAuditEvent({
      action: 'PREDICTION_REVIEW',
      user: req.user,
      req,
      resourceType: 'Prediction',
      resourceId: id,
      details: `Doctor updated notes and marked case as ${status}`,
    });

    res.status(200).json({ success: true, data: prediction });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Delete prediction record
// @route   DELETE /api/predictions/:id
// @access  Private (Admin only)
exports.deletePrediction = async (req, res) => {
  try {
    const { id } = req.params;

    if (isUsingMemoryStore()) {
      const index = store.predictions.findIndex((p) => p._id === id);
      if (index === -1) return res.status(404).json({ success: false, message: 'Prediction not found.' });

      store.predictions.splice(index, 1);
      return res.status(200).json({ success: true, message: 'Prediction deleted successfully.' });
    }

    const prediction = await Prediction.findByIdAndDelete(id);
    if (!prediction) {
      return res.status(404).json({ success: false, message: 'Prediction not found.' });
    }

    res.status(200).json({ success: true, message: 'Prediction deleted successfully.' });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};
