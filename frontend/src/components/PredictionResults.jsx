import React, { useState } from 'react';
import { Target, Clock, Cpu, Maximize2, Layers, AlertTriangle } from 'lucide-react';
import ExplainabilityView from './ExplainabilityView';

export default function PredictionResults({ predictionData, previewUrl }) {
  const [activeView, setActiveView] = useState('predictions'); // 'predictions' | 'explainability'

  if (!predictionData) {
    return (
      <div className="glass-card empty-state">
        <Target className="empty-icon" />
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.4rem' }}>
          No Image Analyzed Yet
        </h3>
        <p style={{ maxWidth: '320px', fontSize: '0.85rem' }}>
          Upload a natural or urban scene image, or click one of the sample presets to run real-time inference.
        </p>
      </div>
    );
  }

  const {
    predicted_class,
    confidence,
    confidence_percentage,
    top_predictions,
    inference_time_ms,
    model_info,
    explainability,
    disclaimer
  } = predictionData;

  return (
    <div className="glass-card">
      <div className="card-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <h2 className="card-title">
            <Target size={20} className="text-emerald-400" />
            Prediction Intelligence
          </h2>
        </div>
        <div className="gradcam-view-modes">
          <button
            className={`mode-btn ${activeView === 'predictions' ? 'active' : ''}`}
            onClick={() => setActiveView('predictions')}
          >
            <Layers size={13} style={{ display: 'inline', marginRight: '3px' }} />
            Rankings
          </button>
          <button
            className={`mode-btn ${activeView === 'explainability' ? 'active' : ''}`}
            onClick={() => setActiveView('explainability')}
          >
            Grad-CAM Heatmap
          </button>
        </div>
      </div>

      {activeView === 'predictions' ? (
        <>
          {/* Primary Class Hero */}
          <div className="primary-prediction-hero">
            <div className="prediction-label-group">
              <span className="prediction-caption">Primary Classification</span>
              <h3 className="predicted-title">{predicted_class}</h3>
            </div>
            <div className="confidence-score-badge">
              <div className="confidence-number mono">{confidence_percentage}</div>
              <span className="confidence-caption">Model Probability</span>
            </div>
          </div>

          {/* Performance & Hardware Specs Chips */}
          <div className="meta-chips-grid">
            <div className="meta-chip">
              <span className="meta-chip-title">Latency</span>
              <span className="meta-chip-val mono" style={{ color: '#38bdf8' }}>
                <Clock size={13} style={{ display: 'inline', marginRight: '2px' }} />
                {inference_time_ms} ms
              </span>
            </div>
            <div className="meta-chip">
              <span className="meta-chip-title">Architecture</span>
              <span className="meta-chip-val mono" style={{ fontSize: '0.8rem', color: '#c084fc' }}>
                <Cpu size={13} style={{ display: 'inline', marginRight: '2px' }} />
                {model_info?.type === 'transfer_learning' ? 'MobileNetV2' : 'CNN Baseline'}
              </span>
            </div>
            <div className="meta-chip">
              <span className="meta-chip-title">Input Size</span>
              <span className="meta-chip-val mono" style={{ color: '#fbbf24' }}>
                <Maximize2 size={13} style={{ display: 'inline', marginRight: '2px' }} />
                {model_info?.input_resolution || '224x224'}
              </span>
            </div>
          </div>

          {/* Top-K Confidence Bars */}
          <div className="confidence-bars-list">
            <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
              Top Predicted Classes Distribution:
            </div>
            {top_predictions.map((item) => (
              <div key={item.class_id} className="bar-row">
                <div className="bar-meta">
                  <span className="bar-class-name">
                    <span className="rank-badge mono">#{item.rank}</span>
                    {item.class_name}
                  </span>
                  <span className="mono" style={{ color: item.rank === 1 ? 'var(--accent-emerald)' : 'var(--text-secondary)' }}>
                    {item.confidence_percentage}
                  </span>
                </div>
                <div className="bar-track">
                  <div
                    className={`bar-fill ${item.rank === 1 ? 'rank-1' : ''}`}
                    style={{ width: `${Math.max(item.confidence * 100, 2)}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>

          <div className="disclaimer-box">
            <AlertTriangle size={13} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'text-bottom' }} />
            {disclaimer}
          </div>
        </>
      ) : (
        <ExplainabilityView
          explainability={explainability}
          previewUrl={previewUrl}
          predictedClass={predicted_class}
        />
      )}
    </div>
  );
}
