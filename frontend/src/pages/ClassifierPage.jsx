import React from 'react';
import ImageUploader from '../components/ImageUploader';
import PredictionResults from '../components/PredictionResults';

export default function ClassifierPage({
  onAnalyze,
  onSampleSelect,
  isLoading,
  selectedFile,
  previewUrl,
  predictionData,
  onClear,
  sampleImagesList,
  error
}) {
  return (
    <div>
      <div className="hero-section">
        <div className="hero-badge">
          <span>VisionAI Deep Learning Platform</span>
        </div>
        <h1 className="hero-title">
          Intelligent Image Recognition <br />
          <span className="hero-gradient">&amp; Visual Feature Intelligence</span>
        </h1>
        <p className="hero-subtitle">
          Real-time scene classification powered by MobileNetV2 transfer learning and custom CNN architectures,
          featuring instant Grad-CAM visual attention mapping and empirical test metrics.
        </p>
      </div>

      <div className="dashboard-grid">
        <ImageUploader
          onAnalyze={onAnalyze}
          onSampleSelect={onSampleSelect}
          isLoading={isLoading}
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          onClear={onClear}
          sampleImagesList={sampleImagesList}
          error={error}
        />

        <PredictionResults
          predictionData={predictionData}
          previewUrl={previewUrl}
        />
      </div>
    </div>
  );
}
