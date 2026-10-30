import React, { useState } from 'react';
import { BarChart2, CheckCircle2, TrendingUp, AlertCircle, RefreshCw, Image as ImageIcon } from 'lucide-react';

export default function MetricsDashboard({ metricsData, onRefresh, isLoading }) {
  const [selectedPlot, setSelectedPlot] = useState('confusion_matrix_mobilenetv2.png');

  if (!metricsData || metricsData.status !== 'available' || !metricsData.models) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '3.5rem 1.5rem' }}>
        <BarChart2 size={40} style={{ color: 'var(--text-muted)', margin: '0 auto 1rem', opacity: 0.5 }} />
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.5rem' }}>
          Metrics Not Available Yet
        </h3>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '480px', margin: '0 auto 1.5rem', fontSize: '0.9rem' }}>
          {metricsData?.message || 'Model evaluation has not been completed. Run the evaluation pipeline against the test split to generate test accuracy, F1-scores, and confusion matrices.'}
        </p>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
          <button className="nav-tab-btn active" onClick={onRefresh} disabled={isLoading}>
            <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
            Check for Metrics
          </button>
        </div>
      </div>
    );
  }

  const { models, classes = [] } = metricsData;
  const mobilenet = models.mobilenetv2;
  const baseline = models.baseline_cnn;
  const primaryModel = mobilenet || baseline;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Top Stat Cards */}
      <div className="metrics-stat-grid">
        <div className="stat-card">
          <div className="stat-label">Test Set Accuracy</div>
          <div className="stat-value mono" style={{ color: 'var(--accent-emerald)' }}>
            {(primaryModel.test_accuracy * 100).toFixed(1)}%
          </div>
          <div className="stat-badge">Measured on Unseen Test Split</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Weighted F1-Score</div>
          <div className="stat-value mono" style={{ color: '#38bdf8' }}>
            {primaryModel.weighted_f1.toFixed(3)}
          </div>
          <div className="stat-badge" style={{ color: '#38bdf8' }}>Harmonic Mean (P & R)</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Macro F1-Score</div>
          <div className="stat-value mono" style={{ color: '#c084fc' }}>
            {primaryModel.macro_f1.toFixed(3)}
          </div>
          <div className="stat-badge" style={{ color: '#c084fc' }}>Unweighted Class Average</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Test Cross-Entropy Loss</div>
          <div className="stat-value mono" style={{ color: '#f59e0b' }}>
            {primaryModel.test_loss.toFixed(4)}
          </div>
          <div className="stat-badge" style={{ color: '#f59e0b' }}>Optimization Objective</div>
        </div>
      </div>

      {/* Baseline vs MobileNetV2 Comparative Analysis */}
      {baseline && mobilenet && (
        <div className="glass-card">
          <div className="card-header">
            <h3 className="card-title">
              <TrendingUp size={20} className="text-blue-400" />
              Comparative Architecture Performance (Unseen Test Set)
            </h3>
          </div>
          <div className="confusion-table-wrapper">
            <table className="confusion-table">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left' }}>Model Architecture</th>
                  <th>Test Accuracy</th>
                  <th>Weighted Precision</th>
                  <th>Weighted Recall</th>
                  <th>Weighted F1</th>
                  <th>Test Loss</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ textAlign: 'left', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    Baseline Custom CNN
                  </td>
                  <td className="mono">{(baseline.test_accuracy * 100).toFixed(2)}%</td>
                  <td className="mono">{baseline.weighted_precision.toFixed(3)}</td>
                  <td className="mono">{baseline.weighted_recall.toFixed(3)}</td>
                  <td className="mono">{baseline.weighted_f1.toFixed(3)}</td>
                  <td className="mono">{baseline.test_loss.toFixed(4)}</td>
                </tr>
                <tr style={{ background: 'rgba(59, 130, 246, 0.08)' }}>
                  <td style={{ textAlign: 'left', fontWeight: 700, color: '#60a5fa' }}>
                    MobileNetV2 (Transfer Learning) ⭐
                  </td>
                  <td className="mono" style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>
                    {(mobilenet.test_accuracy * 100).toFixed(2)}%
                  </td>
                  <td className="mono">{mobilenet.weighted_precision.toFixed(3)}</td>
                  <td className="mono">{mobilenet.weighted_recall.toFixed(3)}</td>
                  <td className="mono" style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>
                    {mobilenet.weighted_f1.toFixed(3)}
                  </td>
                  <td className="mono">{mobilenet.test_loss.toFixed(4)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Per-Class Detailed Breakdown */}
      <div className="glass-card">
        <div className="card-header">
          <h3 className="card-title">
            <CheckCircle2 size={20} className="text-emerald-400" />
            Per-Class Classification Metrics ({primaryModel.model_name})
          </h3>
        </div>
        <div className="confusion-table-wrapper">
          <table className="confusion-table">
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Class Category</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Test Support (Samples)</th>
              </tr>
            </thead>
            <tbody>
              {classes.map((cls) => {
                const metric = primaryModel.per_class[cls];
                if (!metric) return null;
                return (
                  <tr key={cls}>
                    <td style={{ textAlign: 'left', fontWeight: 600, textTransform: 'capitalize' }}>
                      {cls}
                    </td>
                    <td className="mono">{(metric.precision * 100).toFixed(1)}%</td>
                    <td className="mono">{(metric.recall * 100).toFixed(1)}%</td>
                    <td className="mono" style={{ color: '#38bdf8', fontWeight: 700 }}>
                      {metric.f1_score.toFixed(3)}
                    </td>
                    <td className="mono">{metric.support}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrix Visualization Table */}
      {primaryModel.confusion_matrix && (
        <div className="glass-card">
          <div className="card-header">
            <h3 className="card-title">
              <BarChart2 size={20} className="text-purple-400" />
              Confusion Matrix ({primaryModel.model_name})
            </h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Rows = Actual Class • Columns = Predicted Class
            </span>
          </div>
          <div className="confusion-table-wrapper">
            <table className="confusion-table">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left' }}>Actual \ Predicted</th>
                  {classes.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {primaryModel.confusion_matrix.map((row, rowIdx) => (
                  <tr key={classes[rowIdx] || rowIdx}>
                    <th style={{ textAlign: 'left' }}>{classes[rowIdx] || `Class ${rowIdx}`}</th>
                    {row.map((val, colIdx) => (
                      <td
                        key={colIdx}
                        className={`mono ${rowIdx === colIdx ? 'cell-diagonal' : ''}`}
                      >
                        {val}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Saved Visual Plots Viewer */}
      <div className="glass-card">
        <div className="card-header">
          <h3 className="card-title">
            <ImageIcon size={20} className="text-cyan-400" />
            Report Visualizations
          </h3>
          <div className="gradcam-view-modes">
            <button
              className={`mode-btn ${selectedPlot === 'confusion_matrix_mobilenetv2.png' ? 'active' : ''}`}
              onClick={() => setSelectedPlot('confusion_matrix_mobilenetv2.png')}
            >
              Confusion Matrix
            </button>
            <button
              className={`mode-btn ${selectedPlot === 'model_comparison.png' ? 'active' : ''}`}
              onClick={() => setSelectedPlot('model_comparison.png')}
            >
              Model Comparison
            </button>
            <button
              className={`mode-btn ${selectedPlot === 'transfer_learning_training_curves.png' ? 'active' : ''}`}
              onClick={() => setSelectedPlot('transfer_learning_training_curves.png')}
            >
              Training Curves
            </button>
          </div>
        </div>
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          background: 'rgba(0, 0, 0, 0.4)',
          borderRadius: '0.75rem',
          padding: '1rem',
          border: '1px solid rgba(255, 255, 255, 0.08)'
        }}>
          <img
            src={`/metrics/plots/${selectedPlot}`}
            alt={selectedPlot}
            style={{ maxWidth: '100%', maxHeight: '420px', objectFit: 'contain', borderRadius: '0.5rem' }}
            onError={(e) => {
              e.target.style.display = 'none';
            }}
          />
        </div>
      </div>
    </div>
  );
}
