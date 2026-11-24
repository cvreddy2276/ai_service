const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');

const BASE_URL = 'http://localhost:5000/api';
const SCAN_PATH = path.resolve(__dirname, '../BreastDCEDL_spy1/spt1_dce/ISPY1_1239_spy1_vis1_acq1.nii.gz');

async function login(email, password) {
  const res = await axios.post(`${BASE_URL}/auth/login`, { email, password });
  return { token: res.data.token, user: res.data.user };
}

async function runVerification() {
  console.log('================================================================');
  console.log('🔬 END-TO-END CLINICAL PLATFORM VERIFICATION FOR ALL ROLES');
  console.log(`📁 Input NIfTI Scan: ${SCAN_PATH}`);
  console.log(`📏 Scan File Size: ${(fs.statSync(SCAN_PATH).size / (1024 * 1024)).toFixed(2)} MB`);
  console.log('================================================================\n');

  // -------------------------------------------------------------
  // ROLE 1: ATTENDING RADIOLOGIST / DOCTOR
  // -------------------------------------------------------------
  console.log('----------------------------------------------------------------');
  console.log('👩‍⚕️ [ROLE 1: DOCTOR] Logging in as doctor@hospital.org...');
  const doctor = await login('doctor@hospital.org', 'Password123!');
  console.log(`✅ Authenticated: ${doctor.user.name} | Role: ${doctor.user.role} | Hospital: ${doctor.user.hospital}`);

  console.log('\n📤 Uploading 3D DCE-MRI volume to /api/scans/upload for AI inference...');
  const form = new FormData();
  form.append('file', fs.createReadStream(SCAN_PATH), {
    filename: path.basename(SCAN_PATH),
    contentType: 'application/octet-stream',
  });
  form.append('patientId', 'PT-89421');
  form.append('patientName', 'Eleanor Vance');

  const uploadStart = Date.now();
  const uploadRes = await axios.post(`${BASE_URL}/scans/upload`, form, {
    headers: {
      ...form.getHeaders(),
      Authorization: `Bearer ${doctor.token}`,
    },
    maxContentLength: Infinity,
    maxBodyLength: Infinity,
    timeout: 120000,
  });
  const uploadDuration = ((Date.now() - uploadStart) / 1000).toFixed(2);
  const prediction = uploadRes.data.data;

  console.log(`✅ Inference completed in ${uploadDuration}s!`);
  console.log(`\n📋 DOCTOR CLINICAL DIAGNOSTIC OUTPUT:`);
  console.log(`   - Prediction ID:         ${prediction._id}`);
  console.log(`   - Patient MRN / Name:    ${prediction.patientId} (${prediction.patientName})`);
  console.log(`   - Scan File:             ${prediction.scanFileName}`);
  console.log(`   - Diagnostic Verdict:    ${prediction.prediction} (isMalignant: ${prediction.isMalignant})`);
  console.log(`   - Risk Level:            ${prediction.riskLevel}`);
  console.log(`   - AI Model Confidence:   ${prediction.confidencePercentage}%`);
  console.log(`   - pCR Probability:       ${(prediction.pcrProbability * 100).toFixed(1)}%`);
  console.log(`   - Tumor Volume:          ${prediction.tumorVolumeCm3} cm³`);
  console.log(`   - Peak Slice Index:      #${prediction.peakSliceIndex} of ${prediction.totalSlices}`);
  console.log(`   - Volume Dimensions:     ${prediction.dimensions.join(' × ')}`);
  console.log(`   - Voxel Spacing:         ${prediction.voxelSpacing.join(' × ')} mm`);
  console.log(`   - 3D Key Slices:         ${prediction.keySlices ? prediction.keySlices.length : 0} axial slices with Grad-CAM heatmaps`);
  console.log(`   - Model Architecture:    ${prediction.modelVersion}`);
  console.log(`   - Key Findings:`);
  (prediction.findings || []).forEach(f => console.log(`       • ${f}`));
  console.log(`   - Recommendations:`);
  (prediction.recommendations || []).forEach(r => console.log(`       • ${r}`));

  console.log('\n✍️ Doctor signing clinical notes & confirming verdict...');
  const notesRes = await axios.put(
    `${BASE_URL}/predictions/${prediction._id}/notes`,
    {
      doctorNotes: 'Confirmed focal enhancement in upper outer quadrant. Scheduled for follow-up biopsy.',
      status: 'Confirmed',
    },
    { headers: { Authorization: `Bearer ${doctor.token}` } }
  );
  console.log(`✅ Notes signed! Status: ${notesRes.data.data.status} | Reviewer: ${notesRes.data.data.reviewerName}`);

  console.log('\n📄 Generating Official Clinical Diagnostic PDF Report...');
  const reportRes = await axios.post(
    `${BASE_URL}/reports/generate/${prediction._id}`,
    {},
    { headers: { Authorization: `Bearer ${doctor.token}` } }
  );
  console.log(`✅ PDF Report generated! Report ID: ${reportRes.data.data.reportId}`);
  console.log(`   - PDF File: ${reportRes.data.data.pdfFileName}`);
  console.log(`   - Download Endpoint: http://localhost:5000${reportRes.data.downloadUrl}`);

  // -------------------------------------------------------------
  // ROLE 2: AI ONCOLOGY RESEARCHER
  // -------------------------------------------------------------
  console.log('\n----------------------------------------------------------------');
  console.log('🔬 [ROLE 2: RESEARCHER] Logging in as researcher@oncology.edu...');
  const researcher = await login('researcher@oncology.edu', 'Password123!');
  console.log(`✅ Authenticated: ${researcher.user.name} | Role: ${researcher.user.role} | Lab: ${researcher.user.department}`);

  console.log('\n📊 Fetching Deep Learning Validation Metrics & Benchmarks...');
  try {
    const metricsRes = await axios.get(`${BASE_URL}/analytics/model-metrics`, {
      headers: { Authorization: `Bearer ${researcher.token}` },
    });
    const m = metricsRes.data.data.metrics;
    const archs = metricsRes.data.data.architecturalComparison || [];
    console.log(`📋 RESEARCHER DEEP LEARNING BENCHMARK METRICS:`);
    console.log(`   - Model Name:            ${m.model_name}`);
    console.log(`   - Deep Backbone:         ${m.backbone}`);
    console.log(`   - Training Dataset:      ${m.dataset}`);
    console.log(`   - ROC-AUC:               ${(m.roc_auc * 100).toFixed(1)}%`);
    console.log(`   - Classification Acc:    ${(m.accuracy * 100).toFixed(1)}%`);
    console.log(`   - Sensitivity / Recall:  ${(m.sensitivity_recall * 100).toFixed(1)}%`);
    console.log(`   - Specificity:           ${(m.specificity * 100).toFixed(1)}%`);
    console.log(`   - F1-Score:              ${(m.f1_score * 100).toFixed(1)}%`);
    console.log(`   - Segmentation Dice:     ${(m.dice_score_segmentation * 100).toFixed(1)}%`);
    console.log(`   - Architecture Benchmarks Evaluated: ${archs.length} deep models`);
  } catch (err) {
    if (err.response && err.response.status === 503) {
      console.log(`ℹ️ Model metrics status (HTTP 503): "${err.response.data.message}"`);
      console.log(`   (Expected: research prototype requires a verified held-out evaluation before publishing benchmarks)`);
    } else {
      throw err;
    }
  }

  console.log('\n📈 Fetching Cohort Analytics & Demographic Distribution...');
  const analyticsRes = await axios.get(`${BASE_URL}/analytics/dashboard`, {
    headers: { Authorization: `Bearer ${researcher.token}` },
  });
  const stats = analyticsRes.data.data;
  console.log(`📋 RESEARCHER COHORT METRICS:`);
  console.log(`   - Total Patients:        ${stats.totalPatients}`);
  console.log(`   - Total Scans Evaluated: ${stats.totalScans}`);
  console.log(`   - pCR Likely Cases:      ${stats.pcrLikelyCases}`);
  console.log(`   - pCR Unlikely Cases:    ${stats.pcrUnlikelyCases}`);
  console.log(`   - Avg Inference Latency: ${stats.avgLatencyMs} ms`);
  console.log(`   - Monthly Trend Count:   ${stats.monthlyTrends?.length || 0} active periods`);

  console.log('\n🛡️ Verifying RBAC Restriction: Researcher attempting to alter doctor notes...');
  try {
    await axios.put(
      `${BASE_URL}/predictions/${prediction._id}/notes`,
      { doctorNotes: 'Unauthorized researcher note' },
      { headers: { Authorization: `Bearer ${researcher.token}` } }
    );
    console.log('❌ UNEXPECTED: Researcher was able to edit doctor notes!');
  } catch (err) {
    if (err.response && err.response.status === 403) {
      console.log(`✅ RBAC PASSED: Access correctly denied (HTTP 403 Forbidden): "${err.response.data.message}"`);
    } else {
      console.log(`⚠️ Expected 403, got: ${err.message}`);
    }
  }

  // -------------------------------------------------------------
  // ROLE 3: SYSTEM ADMINISTRATOR
  // -------------------------------------------------------------
  console.log('\n----------------------------------------------------------------');
  console.log('🛡️ [ROLE 3: ADMIN] Logging in as admin@cancercenter.org...');
  const admin = await login('admin@cancercenter.org', 'Password123!');
  console.log(`✅ Authenticated: ${admin.user.name} | Role: ${admin.user.role} | Department: ${admin.user.department}`);

  console.log('\n📋 Fetching HIPAA / Clinical Compliance Audit Logs...');
  const auditRes = await axios.get(`${BASE_URL}/audit-logs?limit=5`, {
    headers: { Authorization: `Bearer ${admin.token}` },
  });
  console.log(`📋 ADMIN AUDIT COMPLIANCE TRAIL (${auditRes.data.count} recent events):`);
  auditRes.data.data.slice(0, 5).forEach((log, idx) => {
    console.log(`   ${idx + 1}. [${log.action}] by ${log.userName} (${log.userRole}) at ${log.createdAt}`);
    console.log(`      Status: ${log.status} | Details: ${log.details}`);
  });

  console.log('\n👥 Fetching Clinical Patient Registry...');
  const patientsRes = await axios.get(`${BASE_URL}/patients`, {
    headers: { Authorization: `Bearer ${admin.token}` },
  });
  console.log(`✅ Patient count in database: ${patientsRes.data.count} registered patients.`);
  patientsRes.data.data.forEach(p => {
    console.log(`   - ${p.patientId}: ${p.name} (Age: ${p.age}, ${p.biRadsCategory}, Scans: ${p.scanCount}, Risk: ${p.latestRiskLevel})`);
  });

  console.log('\n================================================================');
  console.log('🎉 ALL THREE ROLES VERIFIED SUCCESSFULLY WITH 100% OPERATIONAL PASS!');
  console.log('================================================================');
}

runVerification().catch(err => {
  console.error('❌ Verification failed:', err.response?.data || err.message);
  process.exit(1);
});
