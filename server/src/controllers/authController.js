const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const User = require('../models/User');
const { store } = require('../services/dataStore');
const { isUsingMemoryStore } = require('../config/db');
const { logAuditEvent } = require('../middleware/audit');

const generateToken = (id, email, role) => {
  return jwt.sign(
    { id, email, role },
    process.env.JWT_SECRET || 'super_secure_breast_cancer_dl_jwt_secret_key_2026',
    { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
  );
};

// @desc    Register new user
// @route   POST /api/auth/register
// @access  Public
exports.register = async (req, res) => {
  try {
    const { name, email, password, role = 'doctor', hospital, department } = req.body;

    if (!name || !email || !password) {
      return res.status(400).json({ success: false, message: 'Please provide name, email, and password.' });
    }

    if (isUsingMemoryStore()) {
      const existing = store.users.find((u) => u.email.toLowerCase() === email.toLowerCase());
      if (existing) {
        return res.status(400).json({ success: false, message: 'Email already registered.' });
      }

      const hashedPassword = await bcrypt.hash(password, 10);
      const newUser = {
        _id: `user-${uuidv4().substring(0, 8)}`,
        name,
        email: email.toLowerCase(),
        password: hashedPassword,
        role,
        hospital: hospital || 'Memorial Cancer Institute',
        department: department || 'Oncology & Medical Imaging',
        avatar: '',
        createdAt: new Date().toISOString(),
      };

      store.users.push(newUser);
      const token = generateToken(newUser._id, newUser.email, newUser.role);

      await logAuditEvent({
        action: 'USER_REGISTER',
        user: newUser,
        req,
        details: `Registered new user ${name} with role ${role}`,
      });

      return res.status(201).json({
        success: true,
        token,
        user: {
          id: newUser._id,
          name: newUser.name,
          email: newUser.email,
          role: newUser.role,
          hospital: newUser.hospital,
          department: newUser.department,
        },
      });
    }

    const userExists = await User.findOne({ email });
    if (userExists) {
      return res.status(400).json({ success: false, message: 'Email already registered.' });
    }

    const user = await User.create({
      name,
      email,
      password,
      role,
      hospital,
      department,
    });

    const token = generateToken(user._id, user.email, user.role);

    await logAuditEvent({
      action: 'USER_REGISTER',
      user,
      req,
      details: `Registered new user ${name} with role ${role}`,
    });

    res.status(201).json({
      success: true,
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        hospital: user.hospital,
        department: user.department,
      },
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Authenticate user & get token
// @route   POST /api/auth/login
// @access  Public
exports.login = async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ success: false, message: 'Please provide email and password.' });
    }

    if (isUsingMemoryStore()) {
      const user = store.users.find((u) => u.email.toLowerCase() === email.toLowerCase());
      if (!user) {
        return res.status(401).json({ success: false, message: 'Invalid email or password.' });
      }

      const isMatch = await bcrypt.compare(password, user.password);
      if (!isMatch) {
        return res.status(401).json({ success: false, message: 'Invalid email or password.' });
      }

      const token = generateToken(user._id, user.email, user.role);

      await logAuditEvent({
        action: 'USER_LOGIN',
        user,
        req,
        details: `User ${user.email} successfully logged in as ${user.role}`,
      });

      return res.status(200).json({
        success: true,
        token,
        user: {
          id: user._id,
          name: user.name,
          email: user.email,
          role: user.role,
          hospital: user.hospital,
          department: user.department,
        },
      });
    }

    const user = await User.findOne({ email }).select('+password');
    if (!user) {
      // Fallback check in store
      const memUser = store.users.find((u) => u.email.toLowerCase() === email.toLowerCase());
      if (memUser && (await bcrypt.compare(password, memUser.password))) {
        const token = generateToken(memUser._id, memUser.email, memUser.role);
        return res.status(200).json({
          success: true,
          token,
          user: {
            id: memUser._id,
            name: memUser.name,
            email: memUser.email,
            role: memUser.role,
            hospital: memUser.hospital,
            department: memUser.department,
          },
        });
      }
      return res.status(401).json({ success: false, message: 'Invalid credentials.' });
    }

    const isMatch = await user.matchPassword(password);
    if (!isMatch) {
      return res.status(401).json({ success: false, message: 'Invalid credentials.' });
    }

    const token = generateToken(user._id, user.email, user.role);

    await logAuditEvent({
      action: 'USER_LOGIN',
      user,
      req,
      details: `User ${user.email} logged in.`,
    });

    res.status(200).json({
      success: true,
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        hospital: user.hospital,
        department: user.department,
      },
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Get currently logged in user
// @route   GET /api/auth/me
// @access  Private
exports.getMe = async (req, res) => {
  try {
    const user = req.user;
    res.status(200).json({
      success: true,
      user: {
        id: user._id || user.id,
        name: user.name,
        email: user.email,
        role: user.role,
        hospital: user.hospital,
        department: user.department,
      },
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Forgot password simulation
// @route   POST /api/auth/forgot-password
// @access  Public
exports.forgotPassword = async (req, res) => {
  const { email } = req.body;
  if (!email) {
    return res.status(400).json({ success: false, message: 'Please provide an email address.' });
  }

  res.status(200).json({
    success: true,
    message: `Password reset link and security verification token dispatched to ${email}.`,
  });
};
