const { v4: uuidv4 } = require('uuid');
const Patient = require('../models/Patient');
const { store } = require('../services/dataStore');
const { isUsingMemoryStore } = require('../config/db');
const { logAuditEvent } = require('../middleware/audit');

// @desc    Get all patients with search & filtering
// @route   GET /api/patients
// @access  Private
exports.getPatients = async (req, res) => {
  try {
    const { search, risk, menopausal } = req.query;

    if (isUsingMemoryStore()) {
      let patients = [...store.patients];

      if (search) {
        const query = search.toLowerCase();
        patients = patients.filter(
          (p) =>
            p.name.toLowerCase().includes(query) ||
            p.patientId.toLowerCase().includes(query) ||
            p.biRadsCategory.toLowerCase().includes(query)
        );
      }

      if (risk && risk !== 'all') {
        patients = patients.filter((p) => p.latestRiskLevel.toLowerCase() === risk.toLowerCase());
      }

      if (menopausal && menopausal !== 'all') {
        patients = patients.filter((p) => p.menopausalStatus.toLowerCase() === menopausal.toLowerCase());
      }

      return res.status(200).json({ success: true, count: patients.length, data: patients });
    }

    const filter = {};
    if (search) {
      filter.$or = [
        { name: { $regex: search, $options: 'i' } },
        { patientId: { $regex: search, $options: 'i' } },
        { biRadsCategory: { $regex: search, $options: 'i' } },
      ];
    }
    if (risk && risk !== 'all') {
      filter.latestRiskLevel = risk;
    }
    if (menopausal && menopausal !== 'all') {
      filter.menopausalStatus = menopausal;
    }

    const patients = await Patient.find(filter).sort({ createdAt: -1 });
    res.status(200).json({ success: true, count: patients.length, data: patients });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Get single patient by ID or MRN
// @route   GET /api/patients/:id
// @access  Private
exports.getPatientById = async (req, res) => {
  try {
    const { id } = req.params;

    if (isUsingMemoryStore()) {
      const patient = store.patients.find((p) => p._id === id || p.patientId === id);
      if (!patient) {
        return res.status(404).json({ success: false, message: 'Patient not found.' });
      }

      const predictions = store.predictions.filter((p) => p.patientId === patient.patientId);
      return res.status(200).json({ success: true, data: { ...patient, predictions } });
    }

    const patient = await Patient.findOne({ $or: [{ _id: id }, { patientId: id }] });
    if (!patient) {
      // Fallback
      const memPatient = store.patients.find((p) => p._id === id || p.patientId === id);
      if (memPatient) {
        const predictions = store.predictions.filter((p) => p.patientId === memPatient.patientId);
        return res.status(200).json({ success: true, data: { ...memPatient, predictions } });
      }
      return res.status(404).json({ success: false, message: 'Patient not found.' });
    }

    res.status(200).json({ success: true, data: patient });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Create new patient record
// @route   POST /api/patients
// @access  Private (Doctor / Admin)
exports.createPatient = async (req, res) => {
  try {
    const {
      patientId,
      name,
      age,
      gender = 'Female',
      menopausalStatus = 'Post-menopausal',
      biRadsCategory = 'BI-RADS 4B',
      hrStatus = 'Positive (HR+)',
      her2Status = 'Negative (HER2-)',
      clinicalNotes = '',
    } = req.body;

    if (!name || !age) {
      return res.status(400).json({ success: false, message: 'Please provide patient name and age.' });
    }

    const mrn = patientId || `PT-${Math.floor(10000 + Math.random() * 90000)}`;

    const patientData = {
      patientId: mrn,
      name,
      age: Number(age),
      gender,
      menopausalStatus,
      biRadsCategory,
      hrStatus,
      her2Status,
      primaryDoctorName: req.user?.name || 'Dr. Sarah Mitchell, MD',
      clinicalNotes,
      scanCount: 0,
      latestRiskLevel: 'Unclassified',
      createdAt: new Date().toISOString(),
    };

    if (isUsingMemoryStore()) {
      const existing = store.patients.find((p) => p.patientId === mrn);
      if (existing) {
        return res.status(400).json({ success: false, message: 'Patient with this MRN already exists.' });
      }
      const newPatient = { _id: `pat-${uuidv4().substring(0, 8)}`, ...patientData };
      store.patients.unshift(newPatient);

      await logAuditEvent({
        action: 'PATIENT_CREATE',
        user: req.user,
        req,
        resourceType: 'Patient',
        resourceId: newPatient._id,
        details: `Created patient ${name} (${mrn})`,
      });

      return res.status(201).json({ success: true, data: newPatient });
    }

    const patient = await Patient.create(patientData);

    await logAuditEvent({
      action: 'PATIENT_CREATE',
      user: req.user,
      req,
      resourceType: 'Patient',
      resourceId: patient._id,
      details: `Created patient ${name} (${mrn})`,
    });

    res.status(201).json({ success: true, data: patient });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Update patient details
// @route   PUT /api/patients/:id
// @access  Private (Doctor / Admin)
exports.updatePatient = async (req, res) => {
  try {
    const { id } = req.params;

    if (isUsingMemoryStore()) {
      const index = store.patients.findIndex((p) => p._id === id || p.patientId === id);
      if (index === -1) {
        return res.status(404).json({ success: false, message: 'Patient not found.' });
      }

      store.patients[index] = { ...store.patients[index], ...req.body };

      await logAuditEvent({
        action: 'PATIENT_UPDATE',
        user: req.user,
        req,
        resourceType: 'Patient',
        resourceId: id,
        details: `Updated patient ${store.patients[index].name}`,
      });

      return res.status(200).json({ success: true, data: store.patients[index] });
    }

    const patient = await Patient.findOneAndUpdate({ $or: [{ _id: id }, { patientId: id }] }, req.body, {
      new: true,
      runValidators: true,
    });

    if (!patient) {
      return res.status(404).json({ success: false, message: 'Patient not found.' });
    }

    await logAuditEvent({
      action: 'PATIENT_UPDATE',
      user: req.user,
      req,
      resourceType: 'Patient',
      resourceId: id,
      details: `Updated patient ${patient.name}`,
    });

    res.status(200).json({ success: true, data: patient });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// @desc    Delete patient record
// @route   DELETE /api/patients/:id
// @access  Private (Admin only)
exports.deletePatient = async (req, res) => {
  try {
    const { id } = req.params;

    if (isUsingMemoryStore()) {
      const index = store.patients.findIndex((p) => p._id === id || p.patientId === id);
      if (index === -1) {
        return res.status(404).json({ success: false, message: 'Patient not found.' });
      }

      const deleted = store.patients.splice(index, 1)[0];

      await logAuditEvent({
        action: 'PATIENT_DELETE',
        user: req.user,
        req,
        resourceType: 'Patient',
        resourceId: id,
        details: `Deleted patient ${deleted.name} (${deleted.patientId})`,
      });

      return res.status(200).json({ success: true, message: 'Patient successfully removed.' });
    }

    const patient = await Patient.findOneAndDelete({ $or: [{ _id: id }, { patientId: id }] });
    if (!patient) {
      return res.status(404).json({ success: false, message: 'Patient not found.' });
    }

    await logAuditEvent({
      action: 'PATIENT_DELETE',
      user: req.user,
      req,
      resourceType: 'Patient',
      resourceId: id,
      details: `Deleted patient ${patient.name} (${patient.patientId})`,
    });

    res.status(200).json({ success: true, message: 'Patient successfully removed.' });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};
