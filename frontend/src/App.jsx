import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ClassifierPage from './pages/ClassifierPage';
import MetricsDashboard from './components/MetricsDashboard';
import ModelInfoPage from './pages/ModelInfoPage';
import { checkHealth, getModelInfo, predictImage, predictSample, getMetrics, getSampleImages } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('classifier'); // 'classifier' | 'metrics' | 'architecture'
  const [healthData, setHealthData] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [sampleImagesList, setSampleImagesList] = useState([]);

  // Classifier state
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Initialize data on mount
  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      const [hData, mInfo, sImages] = await Promise.allSettled([
        checkHealth(),
        getModelInfo(),
        getSampleImages()
      ]);
      if (hData.status === 'fulfilled') setHealthData(hData.value);
      if (mInfo.status === 'fulfilled') setModelInfo(mInfo.value);
      if (sImages.status === 'fulfilled') setSampleImagesList(sImages.value.samples || []);
      
      // Attempt fetching metrics silently
      const mRes = await getMetrics().catch(() => null);
      if (mRes) setMetricsData(mRes);
    } catch (e) {
      console.warn('Initial data fetching error:', e);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setPredictionData(null);
    setError(null);
  };

  const handleAnalyze = async (file, shouldSubmit = true) => {
    setError(null);
    setSelectedFile(file);

    // Create local preview if not already set
    if (!previewUrl || previewUrl.startsWith('data:')) {
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
    }

    if (!shouldSubmit) return;

    setIsLoading(true);
    try {
      const result = await predictImage(file, 5, true);
      setPredictionData(result);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Image prediction failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSampleSelect = async (className) => {
    setError(null);
    setIsLoading(true);
    
    // Set sample preview
    const sample = sampleImagesList.find(s => s.class_name === className);
    if (sample?.data_url) {
      setPreviewUrl(sample.data_url);
    }

    try {
      const result = await predictSample(className, 5, true);
      setPredictionData(result);
      if (!sample?.data_url && result.explainability?.overlay_base64) {
        setPreviewUrl(result.explainability.overlay_base64);
      }
    } catch (err) {
      console.error(err);
      setError(err.message || 'Sample prediction failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        healthData={healthData}
        modelInfo={modelInfo}
      />

      <main className="main-wrapper">
        {activeTab === 'classifier' && (
          <ClassifierPage
            onAnalyze={handleAnalyze}
            onSampleSelect={handleSampleSelect}
            isLoading={isLoading}
            selectedFile={selectedFile}
            previewUrl={previewUrl}
            predictionData={predictionData}
            onClear={handleClear}
            sampleImagesList={sampleImagesList}
            error={error}
          />
        )}

        {activeTab === 'metrics' && (
          <MetricsDashboard
            metricsData={metricsData}
            onRefresh={async () => {
              setIsLoading(true);
              try {
                const res = await getMetrics();
                setMetricsData(res);
              } catch (e) {
                console.error(e);
              } finally {
                setIsLoading(false);
              }
            }}
            isLoading={isLoading}
          />
        )}

        {activeTab === 'architecture' && (
          <ModelInfoPage modelInfo={modelInfo} />
        )}
      </main>

      <footer className="footer">
        <p>
          VisionAI — Intelligent Image Recognition &amp; Analysis Platform • Production ML System Architecture
        </p>
        <p style={{ marginTop: '0.35rem', fontSize: '0.75rem' }}>
          Trained on Multi-Class Natural &amp; Urban Scene Benchmark • MobileNetV2 Transfer Learning vs. Custom CNN Baseline
        </p>
      </footer>
    </div>
  );
}
