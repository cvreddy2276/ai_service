const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const Report = require('../models/Report');
const Prediction = require('../models/Prediction');
const Patient = require('../models/Patient');
const { store } = require('../services/dataStore');
const { isUsingMemoryStore } = require('../config/db');
const { generateMedicalReport } = require('../services/pdfService');
const { logAuditEvent } = require('../middleware/audit');

// @desc    Generate clinical PDF report from prediction
// @route   POST /api/reports/generate/:predictionId
// @access  Private
exports.generateReport = async (req, res) => {
  try {
    const { predictionId } = req.params;

    let prediction;
    let patient;

    if (isUsingMemoryStore()) {
      prediction = store.predictions.find((p) => p._id === predictionId);
      if (prediction) {
        patient = store.patients.find((p) => p.patientId === prediction.patientId);
      }
    } else {
      prediction = await Prediction.findById(predictionId);
      if (prediction) {
        patient = await Patient.findOne({ patientId: prediction.patientId });
      } else {
        prediction = store.predictions.find((p) => p._id === predictionId);
        if (prediction) patient = store.patients.find((p) => p.patientId === prediction.patientId);
      }
    }

    if (!prediction) {
      return res.status(404).json({ success: false, message: 'Prediction not found.' });
    }

    // Generate PDF file
    const { fileName, filePath } = await generateMedicalReport(prediction, patient, req.user);

    const reportId = `REP-${uuidv4().substring(0, 8).toUpperCase()}`;
    const reportData = {
      reportId,
      predictionId: prediction._id,
      patientId: prediction.patientId,
      patientName: prediction.patientName,
      generatedByName: req.user?.name || 'Dr. Sarah Mitchell, MD',
      pdfFileName: fileName,
      summaryText: `3D DCE-MRI Assessment: ${prediction.prediction} with ${prediction.confidencePercentage}% confidence.`,
      diagnosticVerdict: prediction.prediction,
      confidenceScore: prediction.confidencePercentage,
      doctorSignature: 'Electronically Verified by Attending Radiologist',
      createdAt: new Date().toISOString(),
    };

    if (isUsingMemoryStore()) {
      const newReport = { _id: `rep-${uuidv4().substring(0, 8)}`, ...reportData };
      store.reports.unshift(newReport);

      await logAuditEvent({
        action: 'REPORT_GENERATED',
        user: req.user,
        req,
        resourceType: 'Report',
        resourceId: newReport.reportId,
        details: `Generated medical report ${newReport.reportId} for patient ${prediction.patientId}`,
      });

      return res.status(201).json({ success: true, data: newReport, downloadUrl: `/api/reports/${fileName}/download` });
    }

    const report = await Report.create(reportData);

    await logAuditEvent({
      action: 'REPORT_GENERATED',
      user: req.user,
      req,
      resourceType: 'Report',
      resourceId: report.reportId,
      details: `Generated medical report ${report.reportId} for patient ${prediction.patientId}`,
    });

    res.status(201).json({ success: true, data: report, downloadUrl: `/api/reports/${fileName}/download` });
  } catch (err) {
    console.error('[ReportController] Report generation failed:', err);
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Download PDF report
// @route   GET /api/reports/:fileName/download
// @access  Public / Token optional for direct browser download
exports.downloadReport = async (req, res) => {
  try {
    const { fileName } = req.params;
    const filePath = path.join(__dirname, '../../reports', fileName);

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ success: false, message: 'Report PDF file not found.' });
    }

    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`);
    const fileStream = fs.createReadStream(filePath);
    fileStream.pipe(res);
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};
