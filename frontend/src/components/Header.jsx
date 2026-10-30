import React from 'react';
import { Eye, Activity, Cpu, BarChart2, BookOpen } from 'lucide-react';

export default function Header({ activeTab, setActiveTab, healthData, modelInfo }) {
  const isHealthy = healthData?.status === 'healthy';
  const isModelReady = healthData?.model_loaded || modelInfo?.is_model_loaded;

  return (
    <header className="navbar">
      <div className="brand">
        <div className="brand-icon">
          <Eye size={22} strokeWidth={2.5} />
        </div>
        <div>
          <span className="brand-title">VisionAI</span>
          <span className="brand-tag">v1.0</span>
        </div>
      </div>

      <nav className="nav-links">
        <button
          className={`nav-tab-btn ${activeTab === 'classifier' ? 'active' : ''}`}
          onClick={() => setActiveTab('classifier')}
        >
          <Activity size={16} />
          Image Classifier
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          <BarChart2 size={16} />
          Model Metrics
        </button>

        <button
          className={`nav-tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
          onClick={() => setActiveTab('architecture')}
        >
          <Cpu size={16} />
          Architecture & Docs
        </button>
      </nav>

      <div>
        <div className={`status-pill ${isHealthy && isModelReady ? '' : 'offline'}`}>
          <span className="status-dot"></span>
          <span>{isModelReady ? (healthData?.active_model || 'Model Online') : 'Standby Mode'}</span>
        </div>
      </div>
    </header>
  );
}
