import React, { useState, useEffect } from 'react';
import { ShieldCheck, Search, Filter, Clock, User, Globe, FileText, CheckCircle2, AlertCircle } from 'lucide-react';
import { auditService } from '../services/api';

export default function AuditLogsPage() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionFilter, setActionFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');

  const fetchLogs = async () => {
    try {
      setLoading(true);
      const res = await auditService.getAuditLogs({
        action: actionFilter,
        status: statusFilter,
        limit: 100,
      });
      setLogs(res.data?.data || []);
    } catch (err) {
      console.warn('Audit logs fetch fallback');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, [actionFilter, statusFilter]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-2xl glass-panel border border-slate-800">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <ShieldCheck className="w-6 h-6 text-emerald-400" />
            Application Audit Logs
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Audit trail for scan uploads, research model outputs, patient-record changes, and report exports. This prototype does not claim HIPAA or GDPR compliance.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <select
            value={actionFilter}
            onChange={(e) => setActionFilter(e.target.value)}
            className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-white focus:outline-none focus:border-brand-500"
          >
            <option value="all">All Actions</option>
            <option value="SCAN_UPLOAD">Scan Uploads</option>
            <option value="PREDICTION_RUN">AI Predictions</option>
            <option value="PREDICTION_REVIEW">Doctor Reviews</option>
            <option value="PATIENT_CREATE">Patient Registrations</option>
            <option value="REPORT_GENERATED">Report Generations</option>
            <option value="USER_LOGIN">User Logins</option>
          </select>

          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-white focus:outline-none focus:border-brand-500"
          >
            <option value="all">All Statuses</option>
            <option value="SUCCESS">Success</option>
            <option value="WARNING">Warning</option>
            <option value="FAILURE">Failure</option>
          </select>
        </div>
      </div>

      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Timestamp (UTC)</th>
                <th className="py-3 px-4">Action Event</th>
                <th className="py-3 px-4">User</th>
                <th className="py-3 px-4">Role</th>
                <th className="py-3 px-4">IP Address</th>
                <th className="py-3 px-4">Resource</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Event Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {logs.map((log) => (
                <tr key={log._id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3 px-4 font-mono text-[11px] text-slate-400">
                    {new Date(log.createdAt).toLocaleString()}
                  </td>
                  <td className="py-3 px-4 font-semibold text-brand-300 font-mono">
                    {log.action}
                  </td>
                  <td className="py-3 px-4 font-medium text-white">{log.userName}</td>
                  <td className="py-3 px-4 capitalize text-slate-400">{log.userRole}</td>
                  <td className="py-3 px-4 font-mono text-slate-400">{log.ipAddress}</td>
                  <td className="py-3 px-4 font-mono text-slate-300">{log.resourceType}</td>
                  <td className="py-3 px-4">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                        log.status === 'SUCCESS'
                          ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                          : 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                      }`}
                    >
                      {log.status}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-slate-400 truncate max-w-[260px]">
                    {log.details}
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
