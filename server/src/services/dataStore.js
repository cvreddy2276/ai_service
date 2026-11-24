const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');

// In-memory app-demo seed storage. Seeded patient profiles are synthetic examples; no predictions are seeded.
const store = {
  users: [],
  patients: [],
  predictions: [],
  reports: [],
  auditLogs: [],
};

const initStore = async () => {
  if (store.users.length > 0) return;

  const hashedPassword = await bcrypt.hash('Password123!', 10);

  // 1. Seed Users
  store.users = [
    {
      _id: 'user-001',
      name: 'Dr. Sarah Mitchell, MD',
      email: 'doctor@hospital.org',
      password: hashedPassword,
      role: 'doctor',
      hospital: 'Memorial Sloan Cancer Center',
      department: 'Diagnostic Radiology & Oncology',
      avatar: 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=150&auto=format&fit=crop&q=80',
      createdAt: new Date().toISOString(),
    },
    {
      _id: 'user-002',
      name: 'Prof. David Vance, PhD',
      email: 'researcher@oncology.edu',
      password: hashedPassword,
      role: 'researcher',
      hospital: 'National Cancer Research Institute',
      department: 'Computational Oncology & AI Labs',
      avatar: 'https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=150&auto=format&fit=crop&q=80',
      createdAt: new Date().toISOString(),
    },
    {
      _id: 'user-003',
      name: 'Elena Rostova (Admin)',
      email: 'admin@cancercenter.org',
      password: hashedPassword,
      role: 'admin',
      hospital: 'Global Health Imaging Consortium',
      department: 'Medical Systems Administration',
      avatar: 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=150&auto=format&fit=crop&q=80',
      createdAt: new Date().toISOString(),
    },
  ];

  // 2. Seed Patients
  store.patients = [
    {
      _id: 'pat-1001',
      patientId: 'PT-89421',
      name: 'Eleanor Vance',
      age: 52,
      gender: 'Female',
      menopausalStatus: 'Post-menopausal',
      biRadsCategory: 'BI-RADS 4C',
      hrStatus: 'Positive (HR+)',
      her2Status: 'Negative (HER2-)',
      primaryDoctorName: 'Dr. Sarah Mitchell, MD',
      clinicalNotes: 'Palpable upper outer quadrant mass left breast. Core biopsy pending.',
      scanCount: 3,
      latestRiskLevel: 'High',
      createdAt: new Date(Date.now() - 30 * 24 * 3600 * 1000).toISOString(),
    },
    {
      _id: 'pat-1002',
      patientId: 'PT-78204',
      name: 'Margaret Chen',
      age: 46,
      gender: 'Female',
      menopausalStatus: 'Peri-menopausal',
      biRadsCategory: 'BI-RADS 4B',
      hrStatus: 'Positive (HR+)',
      her2Status: 'Positive (HER2+)',
      primaryDoctorName: 'Dr. Sarah Mitchell, MD',
      clinicalNotes: 'Undergoing neoadjuvant chemotherapy cycle 2 monitoring.',
      scanCount: 2,
      latestRiskLevel: 'Critical',
      createdAt: new Date(Date.now() - 20 * 24 * 3600 * 1000).toISOString(),
    },
    {
      _id: 'pat-1003',
      patientId: 'PT-65119',
      name: 'Sophia Rodriguez',
      age: 39,
      gender: 'Female',
      menopausalStatus: 'Pre-menopausal',
      biRadsCategory: 'BI-RADS 3',
      hrStatus: 'Pending',
      her2Status: 'Pending',
      primaryDoctorName: 'Dr. Sarah Mitchell, MD',
      clinicalNotes: 'Dense fibroglandular tissue, interval baseline screening.',
      scanCount: 1,
      latestRiskLevel: 'Low',
      createdAt: new Date(Date.now() - 10 * 24 * 3600 * 1000).toISOString(),
    },
    {
      _id: 'pat-1004',
      patientId: 'PT-91302',
      name: 'Clara Oswald',
      age: 58,
      gender: 'Female',
      menopausalStatus: 'Post-menopausal',
      biRadsCategory: 'BI-RADS 2',
      hrStatus: 'Negative (HR-)',
      her2Status: 'Negative (HER2-)',
      primaryDoctorName: 'Dr. Sarah Mitchell, MD',
      clinicalNotes: 'Annual routine surveillance post benign fibroadenoma resection.',
      scanCount: 1,
      latestRiskLevel: 'Low',
      createdAt: new Date(Date.now() - 5 * 24 * 3600 * 1000).toISOString(),
    },
  ];

  // No prediction outputs are pre-seeded; saved outputs must come from the configured model service.
  store.predictions = [];

  // 4. Seed Audit Logs
  store.auditLogs = [
    {
      _id: 'log-001',
      action: 'USER_LOGIN',
      userId: 'user-001',
      userName: 'Dr. Sarah Mitchell, MD',
      userRole: 'doctor',
      ipAddress: '192.168.1.105',
      resourceType: 'Auth',
      resourceId: 'user-001',
      status: 'SUCCESS',
      details: 'Physician authenticated via two-factor session token.',
      createdAt: new Date(Date.now() - 3 * 3600 * 1000).toISOString(),
    },
    {
      _id: 'log-002',
      action: 'PREDICTION_RUN',
      userId: 'user-001',
      userName: 'Dr. Sarah Mitchell, MD',
      userRole: 'doctor',
      ipAddress: '192.168.1.105',
      resourceType: 'Scan',
      resourceId: 'pred-2001',
      status: 'SUCCESS',
      details: 'Inference executed on scan ISPY1_Chen_DCE_Subtraction_Vis2.nii.gz via 3D CNN.',
      createdAt: new Date(Date.now() - 2 * 3600 * 1000).toISOString(),
    },
    {
      _id: 'log-003',
      action: 'PATIENT_CREATE',
      userId: 'user-003',
      userName: 'Elena Rostova (Admin)',
      userRole: 'admin',
      ipAddress: '192.168.1.20',
      resourceType: 'Patient',
      resourceId: 'pat-1004',
      status: 'SUCCESS',
      details: 'Registered patient Clara Oswald (MRN: PT-91302).',
      createdAt: new Date(Date.now() - 24 * 3600 * 1000).toISOString(),
    },
  ];

  console.log('[DataStore] Seed data successfully initialized.');
};

initStore();

module.exports = { store, initStore };
