const { v4: uuidv4 } = require('uuid');
const AuditLog = require('../models/AuditLog');
const { store } = require('../services/dataStore');
const { isUsingMemoryStore } = require('../config/db');

const logAuditEvent = async ({
  action,
  user,
  req,
  resourceType = 'System',
  resourceId = '',
  status = 'SUCCESS',
  details = '',
}) => {
  try {
    const ipAddress = req?.headers['x-forwarded-for'] || req?.socket?.remoteAddress || '127.0.0.1';
    const logData = {
      action,
      userId: user?._id || user?.id || 'anonymous',
      userName: user?.name || 'Anonymous User',
      userRole: user?.role || 'guest',
      ipAddress: String(ipAddress),
      resourceType,
      resourceId: String(resourceId),
      status,
      details,
      createdAt: new Date().toISOString(),
    };

    if (isUsingMemoryStore()) {
      store.auditLogs.unshift({ _id: `log-${uuidv4().substring(0, 8)}`, ...logData });
    } else {
      await AuditLog.create(logData);
    }
  } catch (err) {
    console.error('[AuditLog] Failed to record audit log:', err.message);
  }
};

module.exports = { logAuditEvent };
