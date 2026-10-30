import React, { useState } from 'react';
import { Eye, Info, AlertCircle } from 'lucide-react';

export default function ExplainabilityView({ explainability, previewUrl, predictedClass }) {
  const [showOriginal, setShowOriginal] = useState(false);

  if (!explainability || explainability.status !== 'success' || !explainability.overlay_base64) {
    return (
      <div style={{ padding: '2rem 1rem', textAlign: 'center' }}>
        <AlertCircle size={32} style={{ color: 'var(--accent-amber)', margin: '0 auto 0.75rem' }} />
        <h4 style={{ color: '#ffffff', fontWeight: 700, marginBottom: '0.4rem' }}>
          Grad-CAM Heatmap Unavailable
        </h4>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '380px', margin: '0 auto' }}>
          {explainability?.explanation || 'Feature activation mapping could not be generated for this architecture.'}
        </p>
      </div>
    );
  }

  return (
    <div className="gradcam-card" style={{ marginTop: 0, borderTop: 'none', paddingTop: 0 }}>
      <div className="gradcam-toggle-header">
        <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#93c5fd', textTransform: 'uppercase' }}>
          Gradient-Weighted Class Activation (Grad-CAM)
        </div>
        <div className="gradcam-view-modes">
          <button
            className={`mode-btn ${!showOriginal ? 'active' : ''}`}
            onClick={() => setShowOriginal(false)}
          >
            Heatmap Overlay
          </button>
          <button
            className={`mode-btn ${showOriginal ? 'active' : ''}`}
            onClick={() => setShowOriginal(true)}
          >
            Original
          </button>
        </div>
      </div>

      <div className="gradcam-image-container">
        <img
          src={showOriginal ? previewUrl : explainability.overlay_base64}
          alt={showOriginal ? 'Original input' : 'Grad-CAM Activation Heatmap'}
          className="gradcam-img"
        />
      </div>

      {/* Colormap Legend */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.4rem 0.75rem',
        background: 'rgba(0, 0, 0, 0.4)',
        borderRadius: '0.4rem',
        marginBottom: '0.75rem',
        fontSize: '0.75rem',
        color: 'var(--text-muted)'
      }}>
        <span>Low Attention (Cool Blue)</span>
        <div style={{
          width: '120px',
          height: '6px',
          borderRadius: '3px',
          background: 'linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)'
        }}></div>
        <span>High Attention (Warm Red)</span>
      </div>

      <div className="explain-text-box">
        <Info size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'text-bottom', color: '#60a5fa' }} />
        {explainability.explanation}
      </div>

      <div className="disclaimer-box">
        {explainability.disclaimer}
      </div>
    </div>
  );
}
