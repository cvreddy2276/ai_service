const jwt = require('jsonwebtoken');
const User = require('../models/User');
const { store } = require('../services/dataStore');
const { isUsingMemoryStore } = require('../config/db');

// Verify JWT token in header Authorization: Bearer <token>
const protect = async (req, res, next) => {
  let token;

  if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
    token = req.headers.authorization.split(' ')[1];
  }

  if (!token) {
    return res.status(401).json({
      success: false,
      message: 'Not authorized to access this route. No token provided.',
    });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'super_secure_breast_cancer_dl_jwt_secret_key_2026');

    if (isUsingMemoryStore()) {
      const user = store.users.find((u) => u._id === decoded.id || u.email === decoded.email);
      if (!user) {
        return res.status(401).json({ success: false, message: 'User not found in store.' });
      }
      req.user = user;
    } else {
      const user = await User.findById(decoded.id);
      if (!user) {
        // Fallback to store if DB switch
        const memUser = store.users.find((u) => u._id === decoded.id || u.email === decoded.email);
        if (!memUser) {
          return res.status(401).json({ success: false, message: 'User not found.' });
        }
        req.user = memUser;
      } else {
        req.user = user;
      }
    }

    next();
  } catch (err) {
    return res.status(401).json({
      success: false,
      message: 'Not authorized. Invalid or expired token.',
    });
  }
};

// Grant access to specific roles
const authorize = (...roles) => {
  return (req, res, next) => {
    if (!req.user || !roles.includes(req.user.role)) {
      return res.status(403).json({
        success: false,
        message: `User role '${req.user?.role || 'unknown'}' is not authorized to access this resource.`,
      });
    }
    next();
  };
};

module.exports = { protect, authorize };
