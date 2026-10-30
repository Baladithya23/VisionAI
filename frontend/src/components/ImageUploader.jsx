import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Sparkles, X, AlertCircle } from 'lucide-react';

const CANONICAL_SAMPLES = [
  { name: 'buildings', label: 'Buildings', icon: '🏢' },
  { name: 'forest', label: 'Forest', icon: '🌲' },
  { name: 'glacier', label: 'Glacier', icon: '❄️' },
  { name: 'mountain', label: 'Mountain', icon: '⛰️' },
  { name: 'sea', label: 'Sea', icon: '🌊' },
  { name: 'street', label: 'Street', icon: '🛣️' },
];

export default function ImageUploader({
  onAnalyze,
  onSampleSelect,
  isLoading,
  selectedFile,
  previewUrl,
  onClear,
  sampleImagesList,
  error
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const processSelectedFile = (file) => {
    // Validate file size (10 MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('Selected image exceeds maximum allowed size (10 MB).');
      return;
    }
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!['.jpg', '.jpeg', '.png', '.webp'].includes(ext)) {
      alert('Supported image formats: JPG, JPEG, PNG, WEBP.');
      return;
    }
    onAnalyze(file, false); // select file without auto-submitting
  };

  return (
    <div className="glass-card">
      <div className="card-header">
        <h2 className="card-title">
          <ImageIcon size={20} className="text-blue-400" />
          Input Image & Ingestion
        </h2>
        <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Max 10 MB • JPG, PNG, WEBP
        </span>
      </div>

      {error && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.15)',
          border: '1px solid rgba(239, 68, 68, 0.4)',
          borderRadius: '0.6rem',
          padding: '0.75rem 1rem',
          marginBottom: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          color: '#fca5a5',
          fontSize: '0.85rem'
        }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {!previewUrl ? (
        <div
          className={`dropzone ${isDragging ? 'drag-active' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
          <UploadCloud className="upload-icon" />
          <p className="upload-instruction">Drop your image here or click to browse</p>
          <p className="upload-sub">Evaluates scene classification across 6 natural & urban domains</p>
        </div>
      ) : (
        <div className="preview-container">
          <img src={previewUrl} alt="Preview" className="preview-img" />
          <button className="remove-btn" onClick={onClear} title="Remove image">
            <X size={14} style={{ display: 'inline', marginRight: '3px' }} />
            Clear
          </button>
        </div>
      )}

      <button
        className="btn-primary"
        disabled={!selectedFile || isLoading}
        onClick={() => onAnalyze(selectedFile, true)}
      >
        {isLoading ? (
          <>
            <span className="status-dot" style={{ background: '#ffffff' }}></span>
            Analyzing Features & Computing Heatmap...
          </>
        ) : (
          <>
            <Sparkles size={18} />
            Analyze Image with VisionAI
          </>
        )}
      </button>

      {/* Instant Sample Selectors */}
      <div className="samples-wrapper">
        <div className="samples-label">⚡ Test with Pre-packaged Sample Scenes:</div>
        <div className="samples-grid">
          {CANONICAL_SAMPLES.map((sample) => {
            const serverSample = sampleImagesList?.find(s => s.class_name === sample.name);
            return (
              <button
                key={sample.name}
                className="sample-thumb-btn"
                title={`Test sample ${sample.label}`}
                onClick={() => onSampleSelect(sample.name)}
                disabled={isLoading}
              >
                {serverSample?.data_url ? (
                  <img src={serverSample.data_url} alt={sample.label} className="sample-thumb-img" />
                ) : (
                  <div style={{ fontSize: '1.5rem', padding: '0.4rem 0' }}>{sample.icon}</div>
                )}
                <span className="sample-thumb-name">{sample.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
