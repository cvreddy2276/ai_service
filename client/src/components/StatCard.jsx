import React from 'react';

export default function StatCard({
  title,
  value,
  subtext,
  icon: Icon,
  trend,
  color = 'brand',
}) {
  const colorStyles = {
    brand: 'text-brand-400 bg-brand-500/10 border-brand-500/20',
    red: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
    green: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    purple: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
  };

  return (
    <div className="glass-card rounded-2xl p-5 border border-slate-800/80 hover:border-slate-700/80 transition-all hover:translate-y-[-2px] shadow-lg">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{title}</p>
          <h3 className="text-2xl sm:text-3xl font-bold text-white mt-1.5 font-mono tracking-tight">{value}</h3>
          {subtext && <p className="text-xs text-slate-400 mt-1">{subtext}</p>}
        </div>
        {Icon && (
          <div className={`p-3 rounded-xl border ${colorStyles[color] || colorStyles.brand}`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>
      {trend && (
        <div className="mt-3 pt-3 border-t border-slate-800/60 flex items-center text-xs">
          <span className={trend.positive ? 'text-emerald-400 font-semibold' : 'text-slate-400'}>
            {trend.value}
          </span>
          <span className="text-slate-400 ml-1.5">{trend.label}</span>
        </div>
      )}
    </div>
  );
}
