import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, Shield, User, LogOut, ChevronDown, Bell, Eye } from 'lucide-react';

export default function Navbar() {
  const { user, role, logout, switchDemoRole } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const getRoleBadgeColor = (r) => {
    switch (r) {
      case 'admin':
        return 'bg-purple-500/10 text-purple-400 border-purple-500/30';
      case 'researcher':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      case 'doctor':
      default:
        return 'bg-brand-500/10 text-brand-400 border-brand-500/30';
    }
  };

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md sticky top-0 z-30 px-4 sm:px-6 flex items-center justify-between">
      {/* Brand & Live Status */}
      <div className="flex items-center gap-3">
        <Link to="/dashboard" className="flex items-center gap-2.5 group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-brand-500/20 group-hover:scale-105 transition-transform">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-bold text-base sm:text-lg tracking-tight text-white flex items-center gap-1.5">
              BreastDCE<span className="text-brand-400 font-extrabold">.AI</span>
            </span>
            <span className="text-[10px] text-slate-400 font-medium tracking-wide block uppercase">
              3D DCE-MRI Deep Learning
            </span>
          </div>
        </Link>

        <div className="hidden md:flex items-center gap-1.5 ml-4 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>FastAPI Engine Ready</span>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-3">
        {/* Quick Demo Role Switcher */}
        <div className="hidden sm:flex items-center gap-1 bg-slate-800/80 p-1 rounded-lg border border-slate-700/60 text-xs text-slate-300">
          <span className="px-2 text-slate-400 font-medium">Role:</span>
          {['doctor', 'researcher', 'admin'].map((r) => (
            <button
              key={r}
              onClick={() => switchDemoRole(r)}
              className={`px-2.5 py-1 rounded capitalize font-medium transition-all ${
                role === r
                  ? 'bg-brand-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
              }`}
            >
              {r}
            </button>
          ))}
        </div>

        {/* User Profile Pill */}
        {user ? (
          <div className="flex items-center gap-3 pl-2 border-l border-slate-800">
            <div className="text-right hidden lg:block">
              <div className="text-sm font-semibold text-slate-200">{user.name}</div>
              <div className="text-xs text-slate-400">{user.hospital || 'Memorial Sloan Cancer Center'}</div>
            </div>

            <div className={`px-2 py-0.5 rounded-full text-xs font-medium border uppercase tracking-wider ${getRoleBadgeColor(role)}`}>
              {role}
            </div>

            <button
              onClick={handleLogout}
              title="Logout"
              className="p-2 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <Link
            to="/login"
            className="px-4 py-2 rounded-lg bg-brand-600 hover:bg-brand-500 text-white font-medium text-sm transition-colors"
          >
            Sign In
          </Link>
        )}
      </div>
    </header>
  );
}
