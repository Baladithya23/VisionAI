import React from 'react';
import { Cpu, Layers, Box, CheckCircle, ShieldCheck, Zap } from 'lucide-react';

export default function ModelInfoPage({ modelInfo }) {
  const isLoaded = modelInfo?.is_model_loaded;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Overview Card */}
      <div className="glass-card">
        <div className="card-header">
          <h2 className="card-title">
            <Cpu size={22} className="text-blue-400" />
            Neural Network Architecture Specifications
          </h2>
          <span className={`status-pill ${isLoaded ? '' : 'offline'}`}>
            <span className="status-dot"></span>
            {isLoaded ? 'Active Model' : 'Standby'}
          </span>
        </div>

        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '1.5rem', lineHeight: '1.6' }}>
          VisionAI implements a multi-tier computer vision architecture engineered for high-throughput edge and server inference.
          The primary production engine utilizes <strong>MobileNetV2</strong> with ImageNet transfer learning weights and inverted residual bottleneck blocks,
          benchmarked alongside a customized <strong>Convolutional Neural Network (CNN)</strong> baseline.
        </p>

        <div className="metrics-stat-grid">
          <div className="stat-card">
            <div className="stat-label">Model Family</div>
            <div className="stat-value mono" style={{ fontSize: '1.25rem', color: '#60a5fa' }}>
              MobileNetV2
            </div>
            <div className="stat-badge">Inverted Residuals</div>
          </div>

          <div className="stat-card">
            <div className="stat-label">Total Parameters</div>
            <div className="stat-value mono" style={{ fontSize: '1.25rem', color: '#34d399' }}>
              {modelInfo?.total_parameters?.toLocaleString() || '~2,500,000'}
            </div>
            <div className="stat-badge">Lightweight Footprint</div>
          </div>

          <div className="stat-card">
            <div className="stat-label">Input Resolution</div>
            <div className="stat-value mono" style={{ fontSize: '1.25rem', color: '#fbbf24' }}>
              224 × 224 × 3
            </div>
            <div className="stat-badge">Standard RGB Normalized</div>
          </div>

          <div className="stat-card">
            <div className="stat-label">Output Classes</div>
            <div className="stat-value mono" style={{ fontSize: '1.25rem', color: '#c084fc' }}>
              6 Categories
            </div>
            <div className="stat-badge">Softmax Probabilities</div>
          </div>
        </div>
      </div>

      {/* Architectural Comparison Grid */}
      <div className="dashboard-grid">
        <div className="glass-card">
          <div className="card-header">
            <h3 className="card-title">
              <Zap size={18} className="text-yellow-400" />
              Transfer Learning (MobileNetV2)
            </h3>
          </div>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            <li style={{ display: 'flex', gap: '0.5rem' }}>
              <CheckCircle size={16} color="var(--accent-emerald)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <span><strong>Pretrained Feature Extractor:</strong> Initialized with 1.4M ImageNet weights, retaining generalized low-level edge, texture, and shape representations.</span>
            </li>
            <li style={{ display: 'flex', gap: '0.5rem' }}>
              <CheckCircle size={16} color="var(--accent-emerald)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <span><strong>Depthwise Separable Convolutions:</strong> Reduces parameter count by ~85% relative to standard 3×3 convolutions with minimal accuracy tradeoff.</span>
            </li>
            <li style={{ display: 'flex', gap: '0.5rem' }}>
              <CheckCircle size={16} color="var(--accent-emerald)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <span><strong>Two-Stage Training:</strong> Stage 1 freezes backbone to train top classification head (Dense 256 + Dropout 0.4); Stage 2 unfreezes top 30 layers with lr=1e-5.</span>
            </li>
          </ul>
        </div>

        <div className="glass-card">
          <div className="card-header">
            <h3 className="card-title">
              <Layers size={18} className="text-blue-400" />
              Baseline Custom CNN
            </h3>
          </div>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            <li style={{ display: 'flex', gap: '0.5rem' }}>
              <CheckCircle size={16} color="var(--accent-blue)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <span><strong>Layer Stack:</strong> 3 Conv2D blocks (32 → 64 → 128 filters, 3×3 kernels) with BatchNormalization and ReLU activations.</span>
            </li>
            <li style={{ display: 'flex', gap: '0.5rem' }}>
              <CheckCircle size={16} color="var(--accent-blue)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <span><strong>Downsampling:</strong> 2×2 MaxPooling layers after each conv block reducing spatial dimensionality from 224×224 down to 28×28.</span>
            </li>
            <li style={{ display: 'flex', gap: '0.5rem' }}>
              <CheckCircle size={16} color="var(--accent-blue)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <span><strong>Scientific Comparison:</strong> Demonstrates why deep transfer learning outperforms standard CNNs trained from scratch on constrained sample sizes.</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Target Classes Catalogue */}
      <div className="glass-card">
        <div className="card-header">
          <h3 className="card-title">
            <Box size={18} className="text-cyan-400" />
            Class Label Taxonomy (6 Domains)
          </h3>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem' }}>
          {(modelInfo?.classes || ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']).map((cls, idx) => (
            <div
              key={cls}
              style={{
                background: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid var(--card-border)',
                borderRadius: '0.65rem',
                padding: '0.75rem 1rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.6rem'
              }}
            >
              <span className="mono rank-badge">#{idx}</span>
              <span style={{ fontWeight: 700, textTransform: 'capitalize' }}>{cls}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
