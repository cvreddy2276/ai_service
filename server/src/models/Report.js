const mongoose = require('mongoose');

const ReportSchema = new mongoose.Schema(
  {
    reportId: {
      type: String,
      required: true,
      unique: true,
    },
    predictionId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Prediction',
      required: true,
    },
    patientId: {
      type: String,
      required: true,
    },
    patientName: {
      type: String,
      required: true,
    },
    generatedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      default: null,
    },
    generatedByName: {
      type: String,
      default: 'Automated Diagnostic Engine',
    },
    pdfFileName: {
      type: String,
      required: true,
    },
    summaryText: {
      type: String,
      default: '',
    },
    diagnosticVerdict: {
      type: String,
      default: 'Inconclusive',
    },
    confidenceScore: {
      type: Number,
      default: 0,
    },
    doctorSignature: {
      type: String,
      default: 'Electronically Verified by Attending Radiologist',
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model('Report', ReportSchema);
