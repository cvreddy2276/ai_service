import React from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  Shield,
  Layers,
  Sparkles,
  BarChart3,
  Cpu,
  ArrowRight,
  CheckCircle2,
  FileCheck,
  Stethoscope,
  Microscope,
  Database,
  ChevronRight,
} from 'lucide-react';

export default function LandingPage() {
  const stats = [];

  const features = [
    {
      icon: Layers,
      title: '3D pCR Response Research Model',
      description:
        'Research prototype for estimating pathological complete response from normalized volumetric input; validation is not established.',
    },
    {
      icon: Sparkles,
      title: 'Explainable AI with 3D Grad-CAM',
      description:
        'Displays Grad-CAM as an illustrative model attribution map; it is not tumor localization or segmentation.',
    },
    {
      icon: Cpu,
      title: 'Volumetry Not Available',
      description:
        'No tumor volume is reported. Grad-CAM attribution is not a validated lesion mask.',
    },
    {
      icon: Shield,
      title: 'Access Controls & Audit Logs',
      description:
        'Includes a patient-disjoint research experiment workflow, role-based access controls, and audit logging. This prototype is not represented as HIPAA compliant.',
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Top Bar */}
      <nav className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-40 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-brand-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-bold text-lg text-white">
              BreastDCE<span className="text-brand-400 font-extrabold">.AI</span>
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Link
            to="/login"
            className="px-4 py-2 rounded-xl text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
          >
            Sign In
          </Link>
          <Link
            to="/dashboard"
            className="px-4 py-2 rounded-xl text-sm font-semibold bg-brand-600 hover:bg-brand-500 text-white shadow-lg shadow-brand-500/20 transition-all flex items-center gap-1.5"
          >
            <span>Launch Platform</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-16 pb-20 px-6 max-w-7xl mx-auto overflow-hidden">
        {/* Ambient Gradient Glows */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-brand-600/15 rounded-full blur-3xl pointer-events-none"></div>

        <div className="text-center max-w-3xl mx-auto relative z-10 space-y-6">
          <div className="mx-auto max-w-2xl rounded-xl border border-amber-500/30 bg-amber-950/30 px-4 py-3 text-sm text-amber-100">
            Research prototype only. Model performance has not been established on a held-out patient evaluation cohort.
          </div>
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-400 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Next-Gen Medical Imaging Deep Learning</span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-tight">
            AI-Powered 3D DCE-MRI <br />
            <span className="bg-gradient-to-r from-brand-400 via-cyan-400 to-teal-300 bg-clip-text text-transparent">
              pCR Response Research
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed font-normal">
            Upload a 3D subtraction NIfTI or 4D DCE scan with pre- and post-contrast phases. This research prototype returns an unvalidated pCR response estimate; it does not measure tumor volume or guide care.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
            <Link
              to="/upload"
              className="px-6 py-3.5 rounded-xl font-semibold bg-brand-600 hover:bg-brand-500 text-white shadow-xl shadow-brand-500/25 transition-all flex items-center gap-2"
            >
              <FileCheck className="w-4 h-4" />
              <span>Submit Research Volume</span>
            </Link>

            <Link
              to="/dashboard"
              className="px-6 py-3.5 rounded-xl font-semibold bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 border border-slate-700 transition-all flex items-center gap-2"
            >
              <span>Explore Research Dashboard</span>
              <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
        </div>

        {/* Live Stats Counter Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-16 max-w-4xl mx-auto">
          {stats.map((stat, i) => (
            <div
              key={i}
              className="glass-card rounded-2xl p-5 text-center border border-slate-800/80 shadow-lg"
            >
              <div className="text-3xl font-extrabold text-white font-mono text-brand-400">
                {stat.value}
              </div>
              <div className="text-xs text-slate-400 mt-1 font-medium">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Deep Learning Pipeline Workflow Section */}
      <section className="py-16 px-6 max-w-7xl mx-auto border-t border-slate-900">
        <div className="text-center max-w-2xl mx-auto mb-12 space-y-3">
          <div className="text-xs font-semibold text-brand-400 uppercase tracking-wider">
            Scientific Pipeline
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-white">
            End-to-End DCE-MRI Deep Learning Pipeline
          </h2>
          <p className="text-sm text-slate-400">
            Uses a research preprocessing workflow with MONAI-based 3D deep learning components. Dataset-specific validation and external replication remain future work.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            {
              step: '01',
              title: 'DCE Subtraction Input',
              desc: 'Uses post-contrast minus pre-contrast intensity as the single model input channel.',
            },
            {
              step: '02',
              title: 'Physical Space Handling',
              desc: 'Canonicalizes image orientation and resamples inference inputs to 1 mm voxel spacing.',
            },
            {
              step: '03',
              title: '3D CNN Research Model',
              desc: 'Estimates pathological complete response from a normalized 3D subtraction volume.',
            },
            {
              step: '04',
              title: 'Grad-CAM Attribution',
              desc: 'Displays a model attribution map; it is not a lesion contour or validated localization.',
            },
          ].map((item, i) => (
            <div
              key={i}
              className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3 relative group hover:border-brand-500/40 transition-colors"
            >
              <span className="text-3xl font-black text-slate-800 group-hover:text-brand-500/30 transition-colors font-mono">
                {item.step}
              </span>
              <h3 className="font-bold text-white text-base">{item.title}</h3>
              <p className="text-xs text-slate-400 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Feature Highlights Grid */}
      <section className="py-16 px-6 max-w-7xl mx-auto border-t border-slate-900">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((f, i) => {
            const Icon = f.icon;
            return (
              <div
                key={i}
                className="glass-card rounded-2xl p-6 border border-slate-800/80 space-y-4 hover:border-slate-700 transition-colors"
              >
                <div className="w-12 h-12 rounded-xl bg-brand-500/10 border border-brand-500/20 flex items-center justify-center text-brand-400">
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-bold text-white">{f.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{f.description}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Research Prototype Summary */}
      <section className="py-16 px-6 max-w-5xl mx-auto text-center space-y-6">
        <div className="p-8 rounded-3xl bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800 space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 text-xs font-semibold">
            <Microscope className="w-3.5 h-3.5" />
            <span>Research & Clinical Capstone Showcase</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-white">
            Research Prototype for Technical Review
          </h2>
          <p className="text-sm text-slate-400 max-w-2xl mx-auto">
            Includes scan upload, saved research outputs, a PDF summary, audit logging, and role-based access. It does not provide validated clinical predictions.
          </p>
          <div className="pt-2">
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-sm transition-all shadow-lg shadow-brand-500/20"
            >
              <span>Explore Dashboard</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-900 py-8 px-6 text-center text-xs text-slate-400">
        <p>Breast DCE-MRI pCR research prototype · no clinical performance or compliance claims</p>
      </footer>
    </div>
  );
}
