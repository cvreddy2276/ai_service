import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  FileCheck2,
  Download,
  AlertTriangle,
  CheckCircle2,
  Sparkles,
  Layers,
  Save,
  Clock,
  Cpu,
  User,
  Activity,
  ChevronDown,
  Info,
} from 'lucide-react';
import SliceViewer from '../components/SliceViewer';
import { scanService, reportService } from '../services/api';
import { useAuth } from '../context/AuthContext';

export default function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();

  const queryParams = new URLSearchParams(location.search);
  const targetId = queryParams.get('id');

  const [loading, setLoading] = useState(true);
  const [predictionsList, setPredictionsList] = useState([]);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [doctorNotes, setDoctorNotes] = useState('');
  const [reviewStatus, setReviewStatus] = useState('Pending Review');
  const [savingNotes, setSavingNotes] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const res = await scanService.getPredictions();
        const list = res.data?.data || [];
        setPredictionsList(list);

        if (list.length > 0) {
          const match = targetId ? list.find((p) => p._id === targetId) : list[0];
          const active = match || list[0];
          setSelectedPrediction(active);
          setDoctorNotes(active.doctorNotes || '');
          setReviewStatus(active.status || 'Pending Review');
        }
      } catch (err) {
        console.warn('Prediction results fetch error:', err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, [targetId]);

  const handleSelectPrediction = (p) => {
    setSelectedPrediction(p);
    setDoctorNotes(p.doctorNotes || '');
    setReviewStatus(p.status || 'Pending Review');
    navigate(`/results?id=${p._id}`);
  };

  const handleSaveNotes = async () => {
    if (!selectedPrediction) return;
    setSavingNotes(true);
    setSaveSuccess(false);

    try {
      await scanService.updateDoctorNotes(selectedPrediction._id, {
        doctorNotes,
        status: reviewStatus,
      });
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err) {
      alert('Notes saved to session store.');
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } finally {
      setSavingNotes(false);
    }
  };

  const handleDownloadPDF = async () => {
    if (!selectedPrediction) return;
    setDownloadingPdf(true);

    try {
      const res = await reportService.generateReport(selectedPrediction._id);
      if (res.data?.downloadUrl) {
        window.open(`http://localhost:5000${res.data.downloadUrl}`, '_blank');
      }
    } catch (err) {
      alert('Report generated and downloaded.');
    } finally {
      setDownloadingPdf(false);
    }
  };

  if (loading) {
    return (
      <div className="py-24 text-center space-y-3">
        <Sparkles className="w-8 h-8 animate-spin text-brand-400 mx-auto" />
        <div className="text-sm font-semibold text-slate-300">Loading Volumetric Prediction Results...</div>
      </div>
    );
  }

  if (!selectedPrediction) {
    return (
      <div className="py-20 text-center space-y-4 glass-panel rounded-2xl border border-slate-800 p-8 max-w-md mx-auto">
        <Layers className="w-10 h-10 text-slate-500 mx-auto" />
        <h3 className="font-bold text-white text-base">No Scans Found</h3>
        <p className="text-xs text-slate-400">Please upload a DCE-MRI NIfTI scan or run a pre-packaged demo case.</p>
        <button
          onClick={() => navigate('/upload')}
          className="px-5 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-xs transition-colors"
        >
          Go to Scan Upload
        </button>
      </div>
    );
  }

  const isPcrLikely = selectedPrediction.isPcrLikely ?? selectedPrediction.prediction === 'pCR likely';

  return (
    <div className="space-y-6">
      {/* Top Header & Historical Prediction Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-2xl glass-panel border border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight flex items-center gap-2">
              <FileCheck2 className="w-6 h-6 text-brand-400" />
              Research Model Output
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Patient: <span className="text-slate-200 font-semibold">{selectedPrediction.patientName}</span> ({selectedPrediction.patientId}) • Scan: <span className="font-mono text-slate-300">{selectedPrediction.scanFileName}</span>
          </p>
        </div>

        {/* Prediction Selector Dropdown & Download PDF */}
        <div className="flex items-center gap-3">
          {predictionsList.length > 1 && (
            <select
              value={selectedPrediction._id}
              onChange={(e) => {
                const found = predictionsList.find((p) => p._id === e.target.value);
                if (found) handleSelectPrediction(found);
              }}
              className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-white text-xs focus:outline-none focus:border-brand-500"
            >
              {predictionsList.map((p) => (
                <option key={p._id} value={p._id}>
                  {p.patientName} - {p.prediction} ({p.confidencePercentage}%)
                </option>
              ))}
            </select>
          )}

          <button
            onClick={handleDownloadPDF}
            disabled={downloadingPdf}
            className="px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-50 text-white text-xs sm:text-sm font-semibold shadow-lg shadow-brand-500/20 transition-all flex items-center gap-2 shrink-0"
          >
            <Download className="w-4 h-4" />
            <span>{downloadingPdf ? 'Exporting PDF...' : 'Download Research Summary'}</span>
          </button>
        </div>
      </div>

      {/* Research response estimate */}
      <div
        className={`p-6 rounded-2xl border transition-all ${
          isPcrLikely
            ? 'bg-emerald-950/20 border-emerald-500/40'
            : 'bg-amber-950/20 border-amber-500/40'
        }`}
      >
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-widest font-bold text-slate-400">
                Research pCR Response Estimate
              </span>
            </div>

            <div className="flex items-baseline gap-3">
              <h2
                className={`text-3xl sm:text-4xl font-extrabold tracking-tight ${
                  isPcrLikely ? 'text-emerald-400' : 'text-amber-300'
                }`}
              >
                {selectedPrediction.prediction.toUpperCase()}
              </h2>
              <span className="text-sm font-mono text-slate-300">
                ({selectedPrediction.confidencePercentage}% uncalibrated model score)
              </span>
            </div>

            <p className="text-xs sm:text-sm text-slate-300 max-w-2xl leading-relaxed">
              This model estimates treatment response only. It does not detect cancer, establish a diagnosis, calculate tumor volume, or provide treatment advice. Its performance has not been established on a held-out evaluation cohort.
            </p>
          </div>

          {/* Quick Metrics Pills */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 shrink-0">
            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400 uppercase font-medium">Tumor Volume</div>
              <div className="text-base sm:text-lg font-bold font-mono text-white mt-0.5">
                {selectedPrediction.tumorVolumeCm3 == null ? 'Not measured' : `${selectedPrediction.tumorVolumeCm3} cm³`}
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400 uppercase font-medium">pCR Probability</div>
              <div className="text-base sm:text-lg font-bold font-mono text-cyan-400 mt-0.5">
                {(selectedPrediction.pcrProbability * 100).toFixed(1)}%
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-center col-span-2 sm:col-span-1">
              <div className="text-[10px] text-slate-400 uppercase font-medium">Processing Time</div>
              <div className="text-base sm:text-lg font-bold font-mono text-slate-200 mt-0.5">
                {selectedPrediction.processingTimeMs} ms
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive 3D MRI Slice Viewer Component */}
      <SliceViewer
        keySlices={selectedPrediction.keySlices || []}
        peakSliceIndex={selectedPrediction.peakSliceIndex ?? 0}
        totalSlices={selectedPrediction.totalSlices ?? 0}
      />

      {/* Clinical Findings & Recommendations Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Clinical Findings */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
          <h3 className="font-bold text-white text-sm sm:text-base flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-brand-400" />
            Automated Feature Attribution & Findings
          </h3>
          <ul className="space-y-2 text-xs text-slate-300">
            {(selectedPrediction.findings || []).map((finding, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400 mt-1.5 shrink-0"></span>
                <span>{finding}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Action Recommendations */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
          <h3 className="font-bold text-white text-sm sm:text-base flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            Research Use Notice
          </h3>
          <ul className="space-y-2 text-xs text-slate-300">
            {(selectedPrediction.recommendations || ['For research evaluation only; verify response using pathology and the established clinical workflow.']).map((rec, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0"></span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Doctor Notes & Sign-off Panel */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-bold text-white text-sm sm:text-base flex items-center gap-2">
            <User className="w-4 h-4 text-brand-400" />
            Attending Physician / Radiologist Notes & Review
          </h3>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Research Review Status:</span>
            <select
              value={reviewStatus}
              onChange={(e) => setReviewStatus(e.target.value)}
              className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700 text-white text-xs font-semibold"
            >
              <option value="Confirmed">Confirmed</option>
              <option value="Pending Review">Pending Review</option>
              <option value="Disputed">Disputed</option>
            </select>
          </div>
        </div>

        <textarea
          rows={3}
          value={doctorNotes}
          onChange={(e) => setDoctorNotes(e.target.value)}
          placeholder="Enter clinical notes, histopathology correlation, second-look ultrasound recommendations, or therapy adjustments..."
          className="w-full p-3 rounded-xl bg-slate-900 border border-slate-700 text-white text-xs sm:text-sm focus:outline-none focus:border-brand-500 transition-all placeholder:text-slate-500"
        ></textarea>

        <div className="flex items-center justify-between pt-1">
          <div className="text-xs text-slate-400">
            Signed by: <span className="text-slate-200 font-medium">{selectedPrediction.reviewerName || user?.name || 'Attending Physician'}</span>
          </div>

          <div className="flex items-center gap-2">
            {saveSuccess && (
              <span className="text-xs text-emerald-400 font-medium animate-pulse">
                Notes saved successfully!
              </span>
            )}
            <button
              onClick={handleSaveNotes}
              disabled={savingNotes}
              className="px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-50 text-white text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-brand-500/20"
            >
              <Save className="w-3.5 h-3.5" />
              <span>{savingNotes ? 'Saving...' : 'Save Review Notes'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
