const User = require('../models/User');
const Patient = require('../models/Patient');
const AuditLog = require('../models/AuditLog');

const seedDatabase = async () => {
  try {
    const userCount = await User.countDocuments();
    if (userCount > 0) {
      console.log(`[Database] Database already initialized (${userCount} users found).`);
      return;
    }

    console.log('[Database] Seeding initial demo clinical data into MongoDB...');

    // Synthetic example profiles for a local interface walkthrough; do not use as clinical records.
    // 1. Users
    // Pre-save hook automatically hashes password
    const doctor = await User.create({
      name: 'Dr. Sarah Mitchell, MD',
      email: 'doctor@hospital.org',
      password: 'Password123!',
      role: 'doctor',
      hospital: 'Memorial Sloan Cancer Center',
      department: 'Diagnostic Radiology & Oncology',
      avatar: 'https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=150&auto=format&fit=crop&q=80',
    });

    const researcher = await User.create({
      name: 'Prof. David Vance, PhD',
      email: 'researcher@oncology.edu',
      password: 'Password123!',
      role: 'researcher',
      hospital: 'National Cancer Research Institute',
      department: 'Computational Oncology & AI Labs',
      avatar: 'https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=150&auto=format&fit=crop&q=80',
    });

    const admin = await User.create({
      name: 'Elena Rostova (Admin)',
      email: 'admin@cancercenter.org',
      password: 'Password123!',
      role: 'admin',
      hospital: 'Global Health Imaging Consortium',
      department: 'Medical Systems Administration',
      avatar: 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=150&auto=format&fit=crop&q=80',
    });

    // 2. Patients
    const patient1 = await Patient.create({
      patientId: 'PT-89421',
      name: 'Eleanor Vance',
      age: 52,
      gender: 'Female',
      menopausalStatus: 'Post-menopausal',
      biRadsCategory: 'BI-RADS 4C',
      hrStatus: 'Positive (HR+)',
      her2Status: 'Negative (HER2-)',
      primaryDoctor: doctor._id,
      primaryDoctorName: 'Dr. Sarah Mitchell, MD',
      clinicalNotes: 'Palpable upper outer quadrant mass left breast. Core biopsy pending.',
      scanCount: 3,
      latestRiskLevel: 'High',
    });

    const patient2 = await Patient.create({
      patientId: 'PT-78204',
      name: 'Margaret Chen',
      age: 46,
      gender: 'Female',
      menopausalStatus: 'Peri-menopausal',
      biRadsCategory: 'BI-RADS 4B',
      hrStatus: 'Positive (HR+)',
      her2Status: 'Positive (HER2+)',
      primaryDoctor: doctor._id,
      primaryDoctorName: 'Dr. Sarah Mitchell, MD',
      clinicalNotes: 'Undergoing neoadjuvant chemotherapy cycle 2 monitoring.',
      scanCount: 2,
      latestRiskLevel: 'Critical',
    });

    const patient3 = await Patient.create({
      patientId: 'PT-65119',
      name: 'Sophia Rodriguez',
      age: 39,
      gender: 'Female',
      menopausalStatus: 'Pre-menopausal',
      biRadsCategory: 'BI-RADS 3',
      hrStatus: 'Pending',
      her2Status: 'Pending',
      primaryDoctor: doctor._id,
      primaryDoctorName: 'Dr. Sarah Mitchell, MD',
      clinicalNotes: 'Dense fibroglandular tissue, interval baseline screening.',
      scanCount: 1,
      latestRiskLevel: 'Low',
    });

    const patient4 = await Patient.create({
      patientId: 'PT-91302',
      name: 'Clara Oswald',
      age: 58,
      gender: 'Female',
      menopausalStatus: 'Post-menopausal',
      biRadsCategory: 'BI-RADS 2',
      hrStatus: 'Negative (HR-)',
      her2Status: 'Negative (HER2-)',
      primaryDoctor: doctor._id,
      primaryDoctorName: 'Dr. Sarah Mitchell, MD',
      clinicalNotes: 'Annual routine surveillance post benign fibroadenoma resection.',
      scanCount: 1,
      latestRiskLevel: 'Low',
    });

    // 3. Do not seed model predictions; all saved outputs must come from inference.

    // 4. Audit Logs
    await AuditLog.create([
      {
        action: 'USER_LOGIN',
        userId: doctor._id.toString(),
        userName: doctor.name,
        userRole: doctor.role,
        ipAddress: '127.0.0.1',
        resourceType: 'Auth',
        resourceId: doctor._id.toString(),
        status: 'SUCCESS',
        details: 'Physician authenticated via clinical session token.',
      },
      {
        action: 'PATIENT_CREATE',
        userId: admin._id.toString(),
        userName: admin.name,
        userRole: admin.role,
        ipAddress: '127.0.0.1',
        resourceType: 'Patient',
        resourceId: patient4._id.toString(),
        status: 'SUCCESS',
        details: 'Registered patient Clara Oswald (MRN: PT-91302).',
      },
    ]);

    console.log('[Database] MongoDB seeded successfully with initial clinical data.');
  } catch (err) {
    console.error('[Database] Seeding error:', err.message);
  }
};

module.exports = { seedDatabase };
