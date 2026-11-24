import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, Lock, Mail, ArrowRight, Shield, AlertCircle, CheckCircle } from 'lucide-react';
import { authService } from '../services/api';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showForgotModal, setShowForgotModal] = useState(false);
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotMessage, setForgotMessage] = useState('');

  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.message || 'Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const fillQuickCredentials = (userEmail) => {
    setEmail(userEmail);
    setPassword('Password123!');
    setError('');
  };

  const handleForgotPassword = async (e) => {
    e.preventDefault();
    try {
      const res = await authService.forgotPassword(forgotEmail);
      setForgotMessage(res.data.message);
    } catch (err) {
      setForgotMessage('Dispatched password reset instructions to your registered email.');
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col justify-center items-center px-4 py-12 relative overflow-hidden">
      {/* Background glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[300px] bg-brand-600/10 rounded-full blur-3xl pointer-events-none"></div>

      <div className="w-full max-w-md relative z-10 space-y-6">
        {/* Brand header */}
        <div className="text-center space-y-2">
          <Link to="/" className="inline-flex items-center gap-2 group">
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-tr from-brand-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-brand-500/20 group-hover:scale-105 transition-transform">
              <Activity className="w-6 h-6 text-white" />
            </div>
          </Link>
          <h2 className="text-2xl font-bold tracking-tight text-white">
            BreastDCE<span className="text-brand-400">.AI</span> Clinical Portal
          </h2>
          <p className="text-xs text-slate-400">
            Sign in to access 3D DCE-MRI inference and patient diagnostic records
          </p>
        </div>

        {/* Quick Demo Credentials Panel */}
        <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2 text-xs">
          <div className="text-[11px] font-semibold text-brand-400 uppercase tracking-wider flex items-center gap-1.5">
            <Shield className="w-3.5 h-3.5" />
            <span>Instant Demo Logins (Password: Password123!)</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <button
              type="button"
              onClick={() => fillQuickCredentials('doctor@hospital.org')}
              className="py-1.5 px-2 rounded-lg bg-slate-800 hover:bg-slate-700/80 text-brand-300 font-medium border border-slate-700 transition-colors text-center"
            >
              Doctor
            </button>
            <button
              type="button"
              onClick={() => fillQuickCredentials('researcher@oncology.edu')}
              className="py-1.5 px-2 rounded-lg bg-slate-800 hover:bg-slate-700/80 text-amber-300 font-medium border border-slate-700 transition-colors text-center"
            >
              Researcher
            </button>
            <button
              type="button"
              onClick={() => fillQuickCredentials('admin@cancercenter.org')}
              className="py-1.5 px-2 rounded-lg bg-slate-800 hover:bg-slate-700/80 text-purple-300 font-medium border border-slate-700 transition-colors text-center"
            >
              Admin
            </button>
          </div>
        </div>

        {/* Login Form Card */}
        <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 shadow-2xl space-y-5">
          {error && (
            <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                Email Address
              </label>
              <div className="relative">
                <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="doctor@hospital.org"
                  className="w-full pl-9 pr-4 py-2.5 rounded-xl bg-slate-900 border border-slate-700 text-white text-sm focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-all"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-slate-300">Password</label>
                <button
                  type="button"
                  onClick={() => setShowForgotModal(true)}
                  className="text-xs text-brand-400 hover:text-brand-300"
                >
                  Forgot password?
                </button>
              </div>
              <div className="relative">
                <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full pl-9 pr-4 py-2.5 rounded-xl bg-slate-900 border border-slate-700 text-white text-sm focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-all"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-50 text-white font-semibold text-sm shadow-lg shadow-brand-500/20 transition-all flex items-center justify-center gap-2"
            >
              <span>{loading ? 'Authenticating...' : 'Sign In to Portal'}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>

          <div className="text-center text-xs text-slate-400 pt-2 border-t border-slate-800">
            Don't have an institutional account?{' '}
            <Link to="/register" className="text-brand-400 font-semibold hover:text-brand-300">
              Register here
            </Link>
          </div>
        </div>
      </div>

      {/* Forgot Password Modal */}
      {showForgotModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-sm w-full p-6 space-y-4 shadow-2xl">
            <h3 className="font-bold text-white text-base">Reset Account Password</h3>
            <p className="text-xs text-slate-400">
              Enter your registered clinical email to receive a secure password reset token.
            </p>

            {forgotMessage ? (
              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs flex items-start gap-2">
                <CheckCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{forgotMessage}</span>
              </div>
            ) : (
              <form onSubmit={handleForgotPassword} className="space-y-3">
                <input
                  type="email"
                  required
                  value={forgotEmail}
                  onChange={(e) => setForgotEmail(e.target.value)}
                  placeholder="your.email@hospital.org"
                  className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-white text-sm focus:outline-none focus:border-brand-500"
                />
                <div className="flex justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => {
                      setShowForgotModal(false);
                      setForgotMessage('');
                    }}
                    className="px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 text-xs"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold"
                  >
                    Send Reset Link
                  </button>
                </div>
              </form>
            )}

            {forgotMessage && (
              <button
                type="button"
                onClick={() => {
                  setShowForgotModal(false);
                  setForgotMessage('');
                }}
                className="w-full py-2 rounded-xl bg-slate-800 text-slate-300 text-xs font-medium"
              >
                Close
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
