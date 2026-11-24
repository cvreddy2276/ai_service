const mongoose = require('mongoose');

const PatientSchema = new mongoose.Schema(
  {
    patientId: {
      type: String,
      required: [true, 'Please provide a unique Patient MRN / ID'],
      unique: true,
      trim: true,
    },
    name: {
      type: String,
      required: [true, 'Please provide patient name'],
      trim: true,
    },
    age: {
      type: Number,
      required: [true, 'Please provide patient age'],
      min: 18,
      max: 120,
    },
    gender: {
      type: String,
      enum: ['Female', 'Male', 'Other'],
      default: 'Female',
    },
    menopausalStatus: {
      type: String,
      enum: ['Pre-menopausal', 'Peri-menopausal', 'Post-menopausal'],
      default: 'Post-menopausal',
    },
    biRadsCategory: {
      type: String,
      enum: ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4A', 'BI-RADS 4B', 'BI-RADS 4C', 'BI-RADS 5', 'BI-RADS 6'],
      default: 'BI-RADS 4B',
    },
    hrStatus: {
      type: String,
      enum: ['Positive (HR+)', 'Negative (HR-)', 'Equivocal', 'Pending'],
      default: 'Positive (HR+)',
    },
    her2Status: {
      type: String,
      enum: ['Positive (HER2+)', 'Negative (HER2-)', 'Equivocal', 'Pending'],
      default: 'Negative (HER2-)',
    },
    primaryDoctor: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      default: null,
    },
    primaryDoctorName: {
      type: String,
      default: 'Dr. Sarah Mitchell, MD',
    },
    clinicalNotes: {
      type: String,
      default: 'Patient scheduled for baseline DCE-MRI tumor staging.',
    },
    scanCount: {
      type: Number,
      default: 0,
    },
    latestRiskLevel: {
      type: String,
      enum: ['Low', 'Moderate', 'High', 'Critical', 'Unclassified'],
      default: 'Unclassified',
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model('Patient', PatientSchema);
