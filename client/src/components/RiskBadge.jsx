import React from 'react';

export default function RiskBadge({ riskLevel = 'Moderate', size = 'md' }) {
  const normalized = riskLevel?.toLowerCase() || 'moderate';

  const configs = {
    critical: {
      label: 'Critical Risk',
      classes: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
      dot: 'bg-rose-500 animate-pulse',
    },
    high: {
      label: 'High Risk',
      classes: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
      dot: 'bg-amber-500',
    },
    moderate: {
      label: 'Moderate Risk',
      classes: 'bg-sky-500/15 text-sky-400 border-sky-500/30',
      dot: 'bg-sky-500',
    },
    low: {
      label: 'Low Risk',
      classes: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
      dot: 'bg-emerald-500',
    },
    unclassified: {
      label: 'Pending Eval',
      classes: 'bg-slate-500/15 text-slate-400 border-slate-500/30',
      dot: 'bg-slate-400',
    },
  };

  const current = configs[normalized] || configs.moderate;
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs';

  return (
    <span className={`inline-flex items-center gap-1.5 font-medium rounded-full border ${current.classes} ${sizeClasses}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${current.dot}`}></span>
      <span>{current.label}</span>
    </span>
  );
}
