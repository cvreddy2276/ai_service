import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  UploadCloud,
  FileCheck,
  Layers,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Cpu,
  ArrowRight,
  Database,
  Info,
} from 'lucide-react';
import { scanService, patientService } from '../services/api';

export default function UploadPage() {
  const navigate = useNavigate();

  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activePhase, setActivePhase] = useState('');
  const [error, setError] = useState('');
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState('PT-78204');
  const [selectedPatientName, setSelectedPatientName] = useState('Margaret Chen');

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const res = await patientService.getPatients();
        if (res.data?.data && res.data.data.length > 0) {
          setPatients(res.data.data);
          setSelectedPatientId(res.data.data[0].patientId);
          setSelectedPatientName(res.data.data[0].name);
        }
      } catch (err) {
        console.warn('Could not fetch patients for upload dropdown');
      }
    };
    fetchPatients();
  }, []);

  const handlePatientChange = (e) => {
    const pId = e.target.value;
    setSelectedPatientId(pId);
    const found = patients.find((p) => p.patientId === pId);
    if (found) setSelectedPatientName(found.name);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (file) => {
    const name = file.name.toLowerCase();
    if (!name.endsWith('.nii') && !name.endsWith('.nii.gz')) {
      setError('Invalid format. Please select a valid NIfTI medical imaging file (.nii or .nii.gz).');
      return;
    }
    setError('');
    setSelectedFile(file);
  };

  const handleUploadAndRun = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setProgress(15);
    setError('');
    setActivePhase('Uploading scan and waiting for the research inference service...');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('patientId', selectedPatientId);
      formData.append('patientName', selectedPatientName);

      const res = await scanService.uploadScan(formData, (event) => {
        const percent = Math.round((event.loaded * 75) / event.total);
        setProgress(Math.max(15, percent));
      });

      setProgress(100);
      setActivePhase('Analysis Complete! Redirecting...');
      setTimeout(() => {
        navigate(`/results?id=${res.data?.data?._id}`);
      }, 700);
    } catch (err) {
      console.error('Scan inference failed:', err);
      setError(err.response?.data?.message || 'Processing failed. Please check microservice connectivity.');
      setUploading(false);
    }
  };

  const handleQuickDemoRun = async (demoId) => {
    setUploading(true);
    setProgress(25);
    setError('');
    setActivePhase('Waiting for the research inference service...');

    try {
      const res = await scanService.predictDemo(demoId, {
        patientId: selectedPatientId,
        patientName: selectedPatientName,
      });

      setProgress(100);
      setActivePhase('Demo Inference Complete! Opening Results...');
      setTimeout(() => {
        navigate(`/results?id=${res.data?.data?._id}`);
      }, 600);
    } catch (err) {
      setError('Demo prediction failed.');
      setUploading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Title */}
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2.5">
          <UploadCloud className="w-6 h-6 text-brand-400" />
          3D DCE-MRI Scan Ingestion & Analysis
        </h1>
        <p className="text-xs sm:text-sm text-slate-400">
          Upload a 3D subtraction NIfTI or 4D DCE NIfTI with pre- and post-contrast phases. Research prototype only.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Patient Assignment Card */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
        <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wide">
          Assign Scan to Patient Record
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <select
              value={selectedPatientId}
              onChange={handlePatientChange}
              disabled={uploading}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-900 border border-slate-700 text-white text-xs sm:text-sm focus:outline-none focus:border-brand-500"
            >
              {patients.map((p) => (
                <option key={p.patientId} value={p.patientId}>
                  {p.name} ({p.patientId}) - {p.biRadsCategory}
                </option>
              ))}
            </select>
          </div>
          <div className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400 flex items-center gap-2">
            <Info className="w-4 h-4 text-cyan-400 shrink-0" />
            <span>Patient assignment is for record keeping; it does not imply model validation or patient-level cross-validation.</span>
          </div>
        </div>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-3xl p-8 sm:p-12 text-center transition-all ${
          dragActive
            ? 'border-brand-400 bg-brand-500/10 scale-[1.01]'
            : 'border-slate-800 bg-slate-900/40 hover:border-slate-700'
        }`}
      >
        <div className="max-w-md mx-auto space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-brand-500/10 border border-brand-500/20 text-brand-400 mx-auto flex items-center justify-center">
            <Layers className="w-8 h-8" />
          </div>

          <div>
            <h3 className="font-bold text-white text-base sm:text-lg">
              Drag & Drop 3D NIfTI Volume Here
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Supports raw compressed <code className="text-brand-300 font-mono">.nii.gz</code> or uncompressed <code className="text-brand-300 font-mono">.nii</code> (up to 100 MB)
            </p>
          </div>

          <div className="flex items-center justify-center gap-3 pt-2">
            <label className="cursor-pointer px-5 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-xs sm:text-sm shadow-lg shadow-brand-500/20 transition-all">
              Browse Local Files
              <input
                type="file"
                accept=".nii,.gz"
                onChange={handleFileInput}
                className="hidden"
                disabled={uploading}
              />
            </label>
          </div>

          {selectedFile && (
            <div className="p-3.5 rounded-xl bg-slate-800/80 border border-slate-700/80 text-left flex items-center justify-between mt-4">
              <div className="flex items-center gap-2.5 truncate">
                <FileCheck className="w-5 h-5 text-emerald-400 shrink-0" />
                <div className="truncate">
                  <div className="text-xs font-semibold text-white truncate">{selectedFile.name}</div>
                  <div className="text-[11px] text-slate-400">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for volumetric inference
                  </div>
                </div>
              </div>

              {!uploading && (
                <button
                  type="button"
                  onClick={handleUploadAndRun}
                  className="px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shrink-0 shadow-sm"
                >
                  Run 3D CNN
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Upload & Inference Progress Indicator */}
      {uploading && (
        <div className="glass-panel p-6 rounded-2xl border border-brand-500/30 space-y-4 glow-blue">
          <div className="flex items-center justify-between text-xs font-semibold">
            <span className="text-brand-400 flex items-center gap-2">
              <Sparkles className="w-4 h-4 animate-spin text-cyan-400" />
              {activePhase || 'Analyzing DCE-MRI Volume...'}
            </span>
            <span className="font-mono text-white">{progress}%</span>
          </div>

          <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
            <div
              className="bg-gradient-to-r from-brand-500 via-cyan-400 to-emerald-400 h-2.5 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Built-in Demo Scans (Instant Test Without Waiting or Uploading Big Files) */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div>
          <h3 className="font-bold text-white text-sm sm:text-base flex items-center gap-2">
            <Database className="w-4 h-4 text-cyan-400" />
            Demo Inputs
          </h3>
          <p className="text-xs text-slate-400">
            Demo inputs are illustrative only and do not measure model accuracy or represent clinical recommendations.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[
            {
              id: 'synth-001',
              title: 'Synthetic example #001',
              badge: 'Synthetic',
              badgeColor: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
              size: 'Synthetic',
              desc: 'Artificial subtraction example. Its output is not a clinical case or accuracy measurement.',
            },
            {
              id: 'synth-002',
              title: 'Synthetic example #002',
              badge: 'Synthetic',
              badgeColor: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
              size: 'Synthetic',
              desc: 'Artificial subtraction example. Its output is not a clinical case or accuracy measurement.',
            },
          ].map((demo) => (
            <div
              key={demo.id}
              className="p-4 rounded-xl bg-slate-900 border border-slate-800 hover:border-slate-700 transition-all flex flex-col justify-between space-y-3"
            >
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${demo.badgeColor}`}>
                    {demo.badge}
                  </span>
                  <span className="text-[11px] font-mono text-slate-400">{demo.size}</span>
                </div>
                <h4 className="font-bold text-white text-xs sm:text-sm">{demo.title}</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed">{demo.desc}</p>
              </div>

              <button
                type="button"
                onClick={() => handleQuickDemoRun(demo.id, demo.title)}
                disabled={uploading}
                className="w-full py-2 rounded-lg bg-slate-800 hover:bg-brand-600 hover:text-white text-brand-400 text-xs font-semibold border border-slate-700 hover:border-brand-500 transition-colors flex items-center justify-center gap-1.5"
              >
                <span>Run Instant Analysis</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
