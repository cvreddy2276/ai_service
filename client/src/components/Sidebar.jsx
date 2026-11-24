import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  UploadCloud,
  FileCheck2,
  Users,
  BarChart3,
  ShieldCheck,
  Compass,
  FileText,
  Sparkles,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Sidebar() {
  const { role } = useAuth();

  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Scan Upload', path: '/upload', icon: UploadCloud },
    { name: 'Results & Grad-CAM', path: '/results', icon: FileCheck2 },
    { name: 'Patient Directory', path: '/patients', icon: Users },
    { name: 'Research Analytics', path: '/analytics', icon: BarChart3 },
    ...(role === 'admin' || role === 'doctor'
      ? [{ name: 'Audit & Compliance', path: '/audit-logs', icon: ShieldCheck }]
      : []),
    { name: 'Public Landing', path: '/', icon: Compass },
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col justify-between shrink-0 min-h-[calc(100vh-4rem)]">
      <div className="p-4 space-y-6">
        <div>
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-3 mb-2">
            Clinical Workflow
          </div>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === '/'}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                      isActive
                        ? 'bg-brand-600/15 text-brand-400 border border-brand-500/20 shadow-sm'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.name}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>

        {/* AI Model Spec Callout */}
        <div className="p-3.5 rounded-xl bg-gradient-to-b from-brand-950/40 to-slate-900 border border-brand-800/40 space-y-2">
          <div className="flex items-center gap-2 text-brand-400 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Research pCR Model</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Research prototype. Held-out performance has not been verified; outputs are not for clinical use.
          </p>
          <div className="flex items-center justify-between text-[11px] pt-1 border-t border-slate-800">
            <span className="text-slate-400">Evaluation status:</span>
            <span className="text-amber-300 font-semibold">Unavailable</span>
          </div>
        </div>
      </div>

      {/* Compliance Footer */}
      <div className="p-4 border-t border-slate-800 text-xs text-slate-400 space-y-1">
        <div className="flex items-center gap-1.5 font-medium text-slate-300">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>Prototype · no compliance claim</span>
        </div>
        <div>Patient-disjoint research split workflow</div>
      </div>
    </aside>
  );
}
