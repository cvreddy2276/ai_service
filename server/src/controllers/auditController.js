const AuditLog = require('../models/AuditLog');
const { store } = require('../services/dataStore');
const { isUsingMemoryStore } = require('../config/db');

// @desc    Get medical compliance audit logs
// @route   GET /api/audit-logs
// @access  Private (Admin / Doctor)
exports.getAuditLogs = async (req, res) => {
  try {
    const { action, status, limit = 50 } = req.query;

    if (isUsingMemoryStore()) {
      let logs = [...store.auditLogs];
      if (action && action !== 'all') {
        logs = logs.filter((l) => l.action.toLowerCase() === action.toLowerCase());
      }
      if (status && status !== 'all') {
        logs = logs.filter((l) => l.status.toLowerCase() === status.toLowerCase());
      }

      return res.status(200).json({ success: true, count: logs.length, data: logs.slice(0, Number(limit)) });
    }

    const filter = {};
    if (action && action !== 'all') filter.action = action;
    if (status && status !== 'all') filter.status = status;

    const logs = await AuditLog.find(filter).sort({ createdAt: -1 }).limit(Number(limit));
    res.status(200).json({ success: true, count: logs.length, data: logs });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};
