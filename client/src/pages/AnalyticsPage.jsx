import React, { useEffect, useState } from 'react';
import { Info } from 'lucide-react';
import { analyticsService } from '../services/api';

const metricLabels = [
  ['accuracy', 'Accuracy'],
  ['roc_auc', 'ROC-AUC'],
  ['pr_auc', 'PR-AUC'],
  ['sensitivity_recall', 'Sensitivity'],
  ['specificity', 'Specificity'],
  ['balanced_accuracy', 'Balanced accuracy'],
  ['f1_score', 'F1 score'],
  ['brier_score', 'Brier score'],
];

function formatMetric(value, key) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return key === 'brier_score' ? value.toFixed(3) : `${(value * 100).toFixed(1)}%`;
}

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState(null);
  const [message, setMessage] = useState('Loading evaluation results…');

  useEffect(() => {
    let active = true;
    analyticsService.getModelMetrics()
      .then((response) => {
        if (!active) return;
        const verified = response.data?.data?.metrics;
        if (verified) {
          setMetrics(verified);
          setMessage('');
        } else {
          setMessage('No verified held-out evaluation results are published.');
        }
      })
      .catch(() => {
        if (active) setMessage('No verified held-out evaluation results are published.');
      });
    return () => { active = false; };
  }, []);

  if (!metrics) {
    return (
      <div className="max-w-3xl mx-auto mt-12 p-8 rounded-2xl border border-amber-500/30 bg-amber-950/20 text-center space-y-3">
        <Info className="w-8 h-8 text-amber-300 mx-auto" />
        <h1 className="text-xl font-bold text-white">Evaluation results unavailable</h1>
        <p className="text-sm text-slate-300">{message}</p>
        <p className="text-xs text-slate-400">The existing four-case dry run is not a performance estimate. Publish metrics only from a reproducible patient-disjoint evaluation.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-white">Research Model Evaluation</h1>
        <p className="mt-1 text-sm text-slate-400">{metrics.model_name || 'pCR response classifier'} · {metrics.dataset || 'Evaluation cohort not specified'}</p>
      </header>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {metricLabels.map(([key, label]) => (
          metrics[key] == null ? null : (
            <div key={key} className="glass-panel p-5 rounded-2xl border border-slate-800">
              <div className="text-xs text-slate-400">{label}</div>
              <div className="mt-2 text-2xl font-bold text-white">{formatMetric(metrics[key], key)}</div>
            </div>
          )
        ))}
      </div>
      <p className="text-xs text-amber-200">Research estimates only. These results do not establish clinical utility or support diagnosis or treatment decisions.</p>
    </div>
  );
}
