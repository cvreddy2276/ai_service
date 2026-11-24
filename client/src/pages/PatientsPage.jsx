import React, { useState, useEffect } from 'react';
import {
  Users,
  Search,
  Plus,
  Filter,
  Eye,
  Edit2,
  Trash2,
  Calendar,
  Activity,
  FileText,
  AlertCircle,
  X,
  Check,
} from 'lucide-react';
import RiskBadge from '../components/RiskBadge';
import { patientService } from '../services/api';
import { useAuth } from '../context/AuthContext';

export default function PatientsPage() {
  const { role } = useAuth();

  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState('all');
  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [editingPatient, setEditingPatient] = useState(null);

  // New Patient Form State
  const [formData, setFormData] = useState({
    patientId: '',
    name: '',
    age: '',
    gender: 'Female',
    menopausalStatus: 'Post-menopausal',
    biRadsCategory: 'BI-RADS 4B',
    hrStatus: 'Positive (HR+)',
    her2Status: 'Negative (HER2-)',
    clinicalNotes: '',
  });

  const fetchPatients = async () => {
    try {
      setLoading(true);
      const res = await patientService.getPatients({ search, risk: riskFilter });
      setPatients(res.data?.data || []);
    } catch (err) {
      console.warn('Patient fetch error:', err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, [search, riskFilter]);

  const handleCreatePatient = async (e) => {
    e.preventDefault();
    try {
      await patientService.createPatient(formData);
      setShowAddModal(false);
      setFormData({
        patientId: '',
        name: '',
        age: '',
        gender: 'Female',
        menopausalStatus: 'Post-menopausal',
        biRadsCategory: 'BI-RADS 4B',
        hrStatus: 'Positive (HR+)',
        her2Status: 'Negative (HER2-)',
        clinicalNotes: '',
      });
      fetchPatients();
    } catch (err) {
      alert(err.response?.data?.message || 'Failed to create patient.');
    }
  };

  const handleUpdatePatient = async (e) => {
    e.preventDefault();
    try {
      await patientService.updatePatient(editingPatient._id, editingPatient);
      setEditingPatient(null);
      fetchPatients();
    } catch (err) {
      alert('Failed to update patient record.');
    }
  };

  const handleDeletePatient = async (id) => {
    if (!window.confirm('Are you sure you want to delete this patient record?')) return;
    try {
      await patientService.deletePatient(id);
      fetchPatients();
    } catch (err) {
      alert('Failed to delete patient.');
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-2xl glass-panel border border-slate-800">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <Users className="w-6 h-6 text-brand-400" />
            Patient Management & Clinical Cohorts
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Maintain longitudinal patient records, BI-RADS scoring, receptor profiles, and imaging history.
          </p>
        </div>

        {(role === 'doctor' || role === 'admin') && (
          <button
            onClick={() => setShowAddModal(true)}
            className="px-4 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-semibold text-xs sm:text-sm shadow-lg shadow-brand-500/20 transition-all flex items-center gap-2 shrink-0"
          >
            <Plus className="w-4 h-4" />
            <span>Add Patient</span>
          </button>
        )}
      </div>

      {/* Filter and Search Bar */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name, MRN, or BI-RADS..."
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-900 border border-slate-700 text-white text-xs sm:text-sm focus:outline-none focus:border-brand-500"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto justify-end">
          <span className="text-xs text-slate-400">Risk Filter:</span>
          <select
            value={riskFilter}
            onChange={(e) => setRiskFilter(e.target.value)}
            className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-white text-xs focus:outline-none focus:border-brand-500"
          >
            <option value="all">All Risk Levels</option>
            <option value="Critical">Critical</option>
            <option value="High">High</option>
            <option value="Moderate">Moderate</option>
            <option value="Low">Low</option>
          </select>
        </div>
      </div>

      {/* Patients Table */}
      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 uppercase tracking-wider text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Patient MRN</th>
                <th className="py-3 px-4">Name</th>
                <th className="py-3 px-4">Age / Gender</th>
                <th className="py-3 px-4">Menopausal Status</th>
                <th className="py-3 px-4">BI-RADS</th>
                <th className="py-3 px-4">Receptors (HR/HER2)</th>
                <th className="py-3 px-4">Scans</th>
                <th className="py-3 px-4">Risk Status</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {patients.map((pat) => (
                <tr key={pat._id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3.5 px-4 font-mono font-semibold text-brand-300">
                    {pat.patientId}
                  </td>
                  <td className="py-3.5 px-4 font-medium text-white">{pat.name}</td>
                  <td className="py-3.5 px-4">
                    {pat.age} Yrs • {pat.gender}
                  </td>
                  <td className="py-3.5 px-4 text-slate-400">{pat.menopausalStatus}</td>
                  <td className="py-3.5 px-4 font-semibold text-slate-200">
                    {pat.biRadsCategory}
                  </td>
                  <td className="py-3.5 px-4 font-mono text-[11px] text-slate-300">
                    {pat.hrStatus} / {pat.her2Status}
                  </td>
                  <td className="py-3.5 px-4 font-mono font-bold text-slate-200">
                    {pat.scanCount || 0}
                  </td>
                  <td className="py-3.5 px-4">
                    <RiskBadge riskLevel={pat.latestRiskLevel} size="sm" />
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      <button
                        onClick={() => setSelectedPatient(pat)}
                        title="View Patient Record"
                        className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                      {(role === 'doctor' || role === 'admin') && (
                        <button
                          onClick={() => setEditingPatient(pat)}
                          title="Edit Patient"
                          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
                        >
                          <Edit2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                      {role === 'admin' && (
                        <button
                          onClick={() => handleDeletePatient(pat._id)}
                          title="Delete Patient"
                          className="p-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add Patient Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <h3 className="font-bold text-white text-base">Register New Patient</h3>
              <button
                onClick={() => setShowAddModal(false)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleCreatePatient} className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">
                    Patient MRN
                  </label>
                  <input
                    type="text"
                    value={formData.patientId}
                    onChange={(e) => setFormData({ ...formData, patientId: e.target.value })}
                    placeholder="PT-89421 (Auto if blank)"
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-white text-xs focus:outline-none focus:border-brand-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">
                    Full Name *
                  </label>
                  <input
                    type="text"
                    required
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="Jane Doe"
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-white text-xs focus:outline-none focus:border-brand-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">Age *</label>
                  <input
                    type="number"
                    required
                    min={18}
                    max={110}
                    value={formData.age}
                    onChange={(e) => setFormData({ ...formData, age: e.target.value })}
                    placeholder="52"
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-white text-xs focus:outline-none focus:border-brand-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">Gender</label>
                  <select
                    value={formData.gender}
                    onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-white text-xs focus:outline-none focus:border-brand-500"
                  >
                    <option value="Female">Female</option>
                    <option value="Male">Male</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">Menopause</label>
                  <select
                    value={formData.menopausalStatus}
                    onChange={(e) => setFormData({ ...formData, menopausalStatus: e.target.value })}
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-white text-xs focus:outline-none focus:border-brand-500"
                  >
                    <option value="Post-menopausal">Post</option>
                    <option value="Peri-menopausal">Peri</option>
                    <option value="Pre-menopausal">Pre</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">BI-RADS</label>
                  <select
                    value={formData.biRadsCategory}
                    onChange={(e) => setFormData({ ...formData, biRadsCategory: e.target.value })}
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-white text-xs focus:outline-none focus:border-brand-500"
                  >
                    <option value="BI-RADS 1">BI-RADS 1 (Negative)</option>
                    <option value="BI-RADS 2">BI-RADS 2 (Benign)</option>
                    <option value="BI-RADS 3">BI-RADS 3 (Probably Benign)</option>
                    <option value="BI-RADS 4A">BI-RADS 4A (Low Suspicion)</option>
                    <option value="BI-RADS 4B">BI-RADS 4B (Moderate Suspicion)</option>
                    <option value="BI-RADS 4C">BI-RADS 4C (High Suspicion)</option>
                    <option value="BI-RADS 5">BI-RADS 5 (Highly Suggestive)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">HR/HER2 Status</label>
                  <select
                    value={`${formData.hrStatus} / ${formData.her2Status}`}
                    onChange={(e) => {
                      const [hr, her2] = e.target.value.split(' / ');
                      setFormData({ ...formData, hrStatus: hr, her2Status: her2 });
                    }}
                    className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-white text-xs focus:outline-none focus:border-brand-500"
                  >
                    <option value="Positive (HR+) / Negative (HER2-)">HR+ / HER2- (Luminal A)</option>
                    <option value="Positive (HR+) / Positive (HER2+)">HR+ / HER2+ (Luminal B)</option>
                    <option value="Negative (HR-) / Positive (HER2+)">HR- / HER2+ (HER2-Enriched)</option>
                    <option value="Negative (HR-) / Negative (HER2-)">HR- / HER2- (Triple Negative)</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Clinical Presentation & Baseline History
                </label>
                <textarea
                  rows={2}
                  value={formData.clinicalNotes}
                  onChange={(e) => setFormData({ ...formData, clinicalNotes: e.target.value })}
                  placeholder="Clinical presentation, palpable mass location, family history..."
                  className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-white text-xs focus:outline-none focus:border-brand-500"
                ></textarea>
              </div>

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold"
                >
                  Save Patient
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Patient Detail Drawer / Modal */}
      {selectedPatient && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div>
                <h3 className="font-bold text-white text-base">{selectedPatient.name}</h3>
                <span className="text-xs font-mono text-brand-300">{selectedPatient.patientId}</span>
              </div>
              <button
                onClick={() => setSelectedPatient(null)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2 text-xs text-slate-300">
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Demographics:</span>
                <span>{selectedPatient.age} Yrs • {selectedPatient.gender} ({selectedPatient.menopausalStatus})</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">BI-RADS Classification:</span>
                <span className="font-semibold text-slate-200">{selectedPatient.biRadsCategory}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Receptors:</span>
                <span className="font-mono">{selectedPatient.hrStatus} | {selectedPatient.her2Status}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Total Scans:</span>
                <span className="font-bold text-white">{selectedPatient.scanCount || 0}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Attending Physician:</span>
                <span>{selectedPatient.primaryDoctorName}</span>
              </div>
              <div className="py-2">
                <span className="text-slate-400 block mb-1">Clinical Notes:</span>
                <p className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 italic">
                  {selectedPatient.clinicalNotes || 'No notes on file.'}
                </p>
              </div>
            </div>

            <button
              onClick={() => setSelectedPatient(null)}
              className="w-full py-2 rounded-xl bg-slate-800 text-slate-300 text-xs font-medium"
            >
              Close Record
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
