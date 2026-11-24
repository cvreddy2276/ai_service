const mongoose = require('mongoose');

const KeySliceSchema = new mongoose.Schema({
  sliceIndex: Number,
  plane: { type: String, default: 'axial' },
  imageBase64: String,
  gradcamBase64: String,
  isPeakAttribution: Boolean,
});

const PredictionSchema = new mongoose.Schema(
  {
    patientId: {
      type: String,
      required: true,
      index: true,
    },
    patientRef: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Patient',
      default: null,
    },
    patientName: {
      type: String,
      default: 'Anonymous Subject',
    },
    scanFileName: {
      type: String,
      required: true,
    },
    scanFilePath: {
      type: String,
      default: '',
    },
    prediction: {
      type: String,
      enum: ['pCR likely', 'pCR unlikely'],
      required: true,
    },
    isPcrLikely: {
      type: Boolean,
      required: true,
    },
    confidence: {
      type: Number,
      required: true,
    },
    confidencePercentage: {
      type: Number,
      required: true,
    },
    riskLevel: {
      type: String,
      enum: ['Unvalidated'],
      required: true,
    },
    pcrProbability: {
      type: Number,
      default: null,
    },
    tumorVolumeCm3: {
      type: Number,
      default: 0,
    },
    peakSliceIndex: {
      type: Number,
      default: 0,
    },
    totalSlices: {
      type: Number,
      default: 0,
    },
    dimensions: [Number],
    voxelSpacing: [Number],
    modelVersion: {
      type: String,
      default: 'pcr-research-v0.1',
    },
    processingTimeMs: {
      type: Number,
      default: 0,
    },
    findings: [String],
    recommendations: [String],
    keySlices: [KeySliceSchema],
    doctorNotes: {
      type: String,
      default: '',
    },
    reviewedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      default: null,
    },
    reviewerName: {
      type: String,
      default: '',
    },
    status: {
      type: String,
      enum: ['Pending Review', 'Confirmed', 'Disputed', 'Archived'],
      default: 'Pending Review',
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model('Prediction', PredictionSchema);
