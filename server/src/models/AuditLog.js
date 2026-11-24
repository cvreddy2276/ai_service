const mongoose = require('mongoose');

const AuditLogSchema = new mongoose.Schema(
  {
    action: {
      type: String,
      required: true,
      enum: [
        'USER_LOGIN',
        'USER_LOGOUT',
        'USER_REGISTER',
        'SCAN_UPLOAD',
        'PREDICTION_RUN',
        'PREDICTION_REVIEW',
        'PATIENT_CREATE',
        'PATIENT_UPDATE',
        'PATIENT_DELETE',
        'REPORT_GENERATED',
        'REPORT_DOWNLOADED',
        'SYSTEM_CONFIG',
      ],
    },
    userId: {
      type: String,
      default: 'anonymous',
    },
    userName: {
      type: String,
      default: 'System User',
    },
    userRole: {
      type: String,
      default: 'system',
    },
    ipAddress: {
      type: String,
      default: '127.0.0.1',
    },
    resourceType: {
      type: String,
      default: 'System',
    },
    resourceId: {
      type: String,
      default: '',
    },
    status: {
      type: String,
      enum: ['SUCCESS', 'FAILURE', 'WARNING'],
      default: 'SUCCESS',
    },
    details: {
      type: String,
      default: '',
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model('AuditLog', AuditLogSchema);
