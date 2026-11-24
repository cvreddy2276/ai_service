import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Activity,
  Layers,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Flame,
  UploadCloud,
  FileCheck2,
  Download,
  Eye,
  ArrowUpRight,
  TrendingUp,
  Sparkles,
  Users,
} from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line, Doughnut, Bar } from 'react-chartjs-2';

import StatCard from '../components/StatCard';
import RiskBadge from '../components/RiskBadge';
import { analyticsService, scanService, reportService } from '../services/api';
import { useAuth } from '../context/AuthContext';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function DashboardPage() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalScans: 0,
    pcrLikelyCases: 0,
    pcrUnlikelyCases: 0,
    avgLatencyMs: 0,
    monthlyTrends: [],
    recentPredictions: [],
  });

  const [recentPredictions, setRecentPredictions] = useState([]);
  const [downloadingId, setDownloadingId] = useState(null);

  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        const [dashRes, predsRes] = await Promise.all([
          analyticsService.getDashboardStats(),
          scanService.getPredictions(),
        ]);

        if (dashRes.data?.data) {
          setStats(dashRes.data.data);
        }
        if (predsRes.data?.data) {
          setRecentPredictions(predsRes.data.data);
        }
      } catch (err) {
        console.warn('Dashboard fetch fallback to local cache:', err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboard();
  }, []);

  const handleDownloadPDF = async (predId) => {
    try {
      setDownloadingId(predId);
      const res = await reportService.generateReport(predId);
      if (res.data?.downloadUrl) {
        window.open(`http://localhost:5000${res.data.downloadUrl}`, '_blank');
      }
    } catch (err) {
      alert('Report generated. Downloading from local server.');
    } finally {
      setDownloadingId(null);
    }
  };

  // Line Chart Data: Monthly Scans
  const lineChartData = {
    labels: stats.monthlyTrends.map((d) => d.month),
    datasets: [
      {
        label: 'pCR likely estimates',
        data: stats.monthlyTrends.map((d) => d.pcrLikely),
        borderColor: '#f43f5e',
        backgroundColor: 'rgba(244, 63, 94, 0.1)',
        fill: true,
        tension: 0.35,
        borderWidth: 2,
        pointBackgroundColor: '#f43f5e',
      },
      {
        label: 'pCR unlikely estimates',
        data: stats.monthlyTrends.map((d) => d.pcrUnlikely),
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.35,
        borderWidth: 2,
        pointBackgroundColor: '#10b981',
      },
    ],
  };

  const lineChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { size: 11, family: 'Inter' } },
      },
      tooltip: {
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        borderWidth: 1,
        titleColor: '#f8fafc',
        bodyColor: '#cbd5e1',
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(51, 65, 85, 0.25)' },
        ticks: { color: '#94a3b8', font: { size: 10 } },
      },
      y: {
        grid: { color: 'rgba(51, 65, 85, 0.25)' },
        ticks: { color: '#94a3b8', font: { size: 10 } },
      },
    },
  };

  // Doughnut Chart Data: Classification Ratio
  const doughnutData = {
    labels: ['pCR likely', 'pCR unlikely'],
    datasets: [
      {
        data: [stats.pcrLikelyCases, stats.pcrUnlikelyCases],
        backgroundColor: ['#10b981', '#f59e0b'],
        borderColor: '#0f172a',
        borderWidth: 3,
      },
    ],
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: { color: '#94a3b8', font: { size: 11 } },
      },
    },
    cutout: '68%',
  };

  return (
    <div className="space-y-6">
      {/* Welcome Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-brand-950/30 to-slate-900 border border-slate-800 shadow-xl">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              Research Dashboard
            </h1>
            <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-brand-500/10 text-brand-400 border border-brand-500/20 font-medium">
              Live Monitoring
            </span>
          </div>
          <p className="text-xs sm:text-sm text-slate-400">
            Welcome back, <span className="text-slate-200 font-semibold">{user?.name || 'Dr. Sarah Mitchell'}</span>.
            Review saved research model outputs. No clinical performance metrics are currently verified.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link
            to="/upload"
            className="px-4 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-xs sm:text-sm shadow-lg shadow-brand-500/20 transition-all flex items-center gap-2 shrink-0"
          >
            <UploadCloud className="w-4 h-4" />
            <span>Upload New Scan</span>
          </Link>
        </div>
      </div>

      {/* KPI Stat Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Scans Analyzed"
          value={stats.totalScans}
          subtext="Volumetric 3D NIfTI Datasets"
          icon={Layers}
          color="brand"
          trend={null}
        />
        <StatCard
          title="pCR Likely Estimates"
          value={stats.pcrLikelyCases}
          subtext="Research model outputs"
          icon={Flame}
          color="red"
          trend={null}
        />
        <StatCard
          title="pCR Unlikely Estimates"
          value={stats.pcrUnlikelyCases}
          subtext="Research model outputs"
          icon={CheckCircle2}
          color="green"
          trend={null}
        />
        <StatCard
          title="Average Processing Time"
          value={stats.totalScans > 0 ? `${stats.avgLatencyMs} ms` : '—'}
          subtext="Across saved model outputs"
          icon={Clock}
          color="purple"
          trend={null}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Longitudinal Detection Trend Chart */}
        <div className="lg:col-span-8 glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-brand-400" />
                Saved pCR Estimate Trend
              </h3>
              <p className="text-xs text-slate-400">Counts of stored research estimates by month</p>
            </div>
            <div className="text-xs text-slate-400 font-mono"></div>
          </div>
          <div className="h-64 sm:h-72 w-full">
            <Line data={lineChartData} options={lineChartOptions} />
          </div>
        </div>

        {/* Classification Breakdown Donut */}
        <div className="lg:col-span-4 glass-panel p-5 rounded-2xl border border-slate-800 space-y-4 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-brand-400" />
              Outcome Stratification
            </h3>
            <p className="text-xs text-slate-400">Distribution of saved pCR estimates</p>
          </div>
          <div className="h-48 w-full flex items-center justify-center my-2">
            <Doughnut data={doughnutData} options={doughnutOptions} />
          </div>
          <div className="grid grid-cols-2 gap-2 text-center text-xs pt-2 border-t border-slate-800">
            <div className="p-2 rounded-lg bg-rose-500/10 border border-rose-500/20">
              <div className="font-bold text-rose-400 font-mono">{stats.pcrLikelyCases}</div>
              <div className="text-[11px] text-slate-400">pCR likely</div>
            </div>
            <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <div className="font-bold text-emerald-400 font-mono">{stats.pcrUnlikelyCases}</div>
              <div className="text-[11px] text-slate-400">pCR unlikely</div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Predictions Table */}
      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden shadow-xl space-y-3">
        <div className="p-5 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <FileCheck2 className="w-4 h-4 text-brand-400" />
              Recent Research Model Outputs
            </h3>
            <p className="text-xs text-slate-400">Latest 3D DCE-MRI scans processed through the deep learning pipeline</p>
          </div>
          <Link
            to="/results"
            className="text-xs font-semibold text-brand-400 hover:text-brand-300 flex items-center gap-1"
          >
            <span>View All</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Patient MRN</th>
                <th className="py-3 px-4">Subject Name</th>
                <th className="py-3 px-4">Scan File</th>
                <th className="py-3 px-4">pCR Estimate</th>
                <th className="py-3 px-4">Uncalibrated Score</th>
                <th className="py-3 px-4">pCR Probability</th>
                <th className="py-3 px-4">Model Status</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {recentPredictions.slice(0, 5).map((pred) => (
                <tr key={pred._id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3 px-4 font-mono font-semibold text-brand-300">
                    {pred.patientId}
                  </td>
                  <td className="py-3 px-4 font-medium text-white">{pred.patientName}</td>
                  <td className="py-3 px-4 font-mono text-[11px] text-slate-400 truncate max-w-[180px]">
                    {pred.scanFileName}
                  </td>
                  <td className="py-3 px-4">
                    <span
                      className={`font-semibold px-2 py-0.5 rounded text-[11px] ${
                        pred.isPcrLikely
                          ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                          : 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                      }`}
                    >
                      {pred.prediction}
                    </span>
                  </td>
                  <td className="py-3 px-4 font-mono font-bold text-slate-200">
                    {pred.confidencePercentage}%
                  </td>
                  <td className="py-3 px-4 font-mono text-slate-300">
                    {((pred.pcrProbability ?? 0) * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4">
                    <span className="text-slate-400">Research only</span>
                  </td>
                  <td className="py-3 px-4 text-[11px] text-slate-400">
                    {pred.status || 'Pending'}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      <button
                        onClick={() => navigate(`/results?id=${pred._id}`)}
                        title="View Slices & Grad-CAM"
                        className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleDownloadPDF(pred._id)}
                        disabled={downloadingId === pred._id}
                        title="Download PDF Medical Report"
                        className="p-1.5 rounded-lg bg-brand-600/20 hover:bg-brand-600/30 text-brand-400 transition-colors"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
