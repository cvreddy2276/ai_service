const PDFDocument = require('pdfkit');
const fs = require('fs');
const path = require('path');

const reportsDir = path.join(__dirname, '../../reports');
if (!fs.existsSync(reportsDir)) {
  fs.mkdirSync(reportsDir, { recursive: true });
}

const generateMedicalReport = async (predictionData, patientData, doctorUser) => {
  return new Promise((resolve, reject) => {
    try {
      const fileName = `Report_${predictionData.patientId}_${Date.now()}.pdf`;
      const filePath = path.join(reportsDir, fileName);

      const doc = new PDFDocument({ margin: 40, size: 'A4' });
      const writeStream = fs.createWriteStream(filePath);
      doc.pipe(writeStream);

      // Colors
      const primaryColor = '#1e3a8a'; // Deep Navy
      const darkColor = '#0f172a';
      const grayColor = '#475569';
      const lightBg = '#f1f5f9';
      const alertColor = predictionData.isPcrLikely ? '#16a34a' : '#d97706';

      // --- Header ---
      doc.rect(40, 40, 515, 60).fill('#0f172a');
      doc.fillColor('#ffffff').fontSize(18).font('Helvetica-Bold').text('BREAST DCE-MRI RESEARCH', 55, 52);
      doc.fontSize(10).font('Helvetica').text('pCR RESPONSE MODEL OUTPUT SUMMARY', 55, 75);

      // --- Title Banner ---
      doc.moveDown(2.5);
      doc.fillColor(primaryColor).fontSize(14).font('Helvetica-Bold').text('RESEARCH MODEL OUTPUT REPORT', { underline: true });
      doc.moveDown(0.5);

      // --- Patient & Study Metadata Table ---
      const tableTop = 150;
      doc.rect(40, tableTop, 515, 70).fill(lightBg);

      doc.fillColor(darkColor).fontSize(9).font('Helvetica-Bold');
      doc.text('PATIENT NAME:', 50, tableTop + 10);
      doc.font('Helvetica').text(patientData?.name || predictionData.patientName || 'Anonymous', 140, tableTop + 10);

      doc.font('Helvetica-Bold').text('PATIENT ID (MRN):', 310, tableTop + 10);
      doc.font('Helvetica').text(predictionData.patientId, 415, tableTop + 10);

      doc.font('Helvetica-Bold').text('AGE / GENDER:', 50, tableTop + 28);
      doc.font('Helvetica').text(`${patientData?.age ?? 'N/A'} Yrs / ${patientData?.gender || 'N/A'}`, 140, tableTop + 28);

      doc.font('Helvetica-Bold').text('BI-RADS CATEGORY:', 310, tableTop + 28);
      doc.font('Helvetica').text(patientData?.biRadsCategory || 'N/A', 415, tableTop + 28);

      doc.font('Helvetica-Bold').text('RECEPTOR STATUS:', 50, tableTop + 46);
      doc.font('Helvetica').text(`${patientData?.hrStatus || 'N/A'} | ${patientData?.her2Status || 'N/A'}`, 140, tableTop + 46);

      doc.font('Helvetica-Bold').text('SCAN FILE:', 310, tableTop + 46);
      doc.font('Helvetica').text(predictionData.scanFileName.substring(0, 24), 415, tableTop + 46);

      // --- AI Verdict Box ---
      const verdictTop = 235;
      doc.rect(40, verdictTop, 515, 65).lineWidth(1.5).strokeColor(alertColor).stroke();

      doc.fillColor(alertColor).fontSize(13).font('Helvetica-Bold').text('pCR RESPONSE ESTIMATE:', 55, verdictTop + 12);
      doc.fontSize(16).text(
        `${predictionData.prediction.toUpperCase()} (${predictionData.confidencePercentage}% CONFIDENCE)`,
        210,
        verdictTop + 10
      );

      doc.fillColor(darkColor).fontSize(10).font('Helvetica-Bold');
      doc.text('MODEL STATUS:', 55, verdictTop + 40);
      doc.fillColor(alertColor).text('RESEARCH ONLY', 130, verdictTop + 40);

      doc.fillColor(darkColor).text('pCR PROBABILITY:', 220, verdictTop + 40);
      doc.font('Helvetica').text(`${((predictionData.pcrProbability || 0) * 100).toFixed(1)}%`, 355, verdictTop + 40);

      doc.font('Helvetica-Bold').text('PEAK SLICE:', 430, verdictTop + 40);
      doc.font('Helvetica').text(`#${predictionData.peakSliceIndex}`, 505, verdictTop + 40);

      // --- Findings Section ---
      doc.moveDown(4.5);
      doc.fillColor(primaryColor).fontSize(11).font('Helvetica-Bold').text('MODEL OUTPUT NOTES');
      doc.moveDown(0.4);

      const findings = predictionData.findings || [];
      findings.forEach((finding) => {
        doc.fillColor(darkColor).fontSize(9).font('Helvetica').text(`•  ${finding}`, { indent: 10 });
        doc.moveDown(0.2);
      });

      // --- Recommendations Section ---
      doc.moveDown(0.6);
      doc.fillColor(primaryColor).fontSize(11).font('Helvetica-Bold').text('RESEARCH USE NOTICE');
      doc.moveDown(0.4);

      const recs = predictionData.recommendations || [];
      recs.forEach((rec) => {
        doc.fillColor(darkColor).fontSize(9).font('Helvetica').text(`•  ${rec}`, { indent: 10 });
        doc.moveDown(0.2);
      });

      // --- Attending Notes & Signature ---
      doc.moveDown(0.8);
      doc.rect(40, doc.y, 515, 65).fill('#f8fafc');
      const noteY = doc.y + 8;
      doc.fillColor(darkColor).fontSize(9).font('Helvetica-Bold').text('ATTENDING PHYSICIAN / RADIOLOGIST NOTES:', 50, noteY);
      doc.font('Helvetica-Oblique').text(
        predictionData.doctorNotes || 'No reviewer notes recorded.',
        50,
        noteY + 16,
        { width: 495 }
      );

      // Footer
      doc.fontSize(8).font('Helvetica').fillColor(grayColor);
      doc.fontSize(8).font('Helvetica-Bold').fillColor('#b91c1c').text(
        'Research prototype only. Not validated for diagnosis, treatment selection, or clinical decisions.',
        40,
        755,
        { align: 'center', width: 515 }
      );
      doc.text(
        `Generated on ${new Date().toUTCString()} | Model: ${predictionData.modelVersion} | Research prototype`,
        40,
        780,
        { align: 'center', width: 515 }
      );

      doc.end();

      writeStream.on('finish', () => {
        resolve({ fileName, filePath });
      });

      writeStream.on('error', (err) => {
        reject(err);
      });
    } catch (err) {
      reject(err);
    }
  });
};

module.exports = { generateMedicalReport };
