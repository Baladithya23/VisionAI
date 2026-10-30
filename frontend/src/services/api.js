/**
 * VisionAI - API Client Service
 */

const API_BASE = '';

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return await res.json();
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`);
  if (!res.ok) throw new Error(`Model info failed: ${res.statusText}`);
  return await res.json();
}

export async function predictImage(file, topK = 5, gradcam = true) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/predict?top_k=${topK}&gradcam=${gradcam}`, {
    method: 'POST',
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || 'Prediction failed. Please check your image.');
  }
  return data;
}

export async function predictSample(className, topK = 5, gradcam = true) {
  const res = await fetch(`${API_BASE}/predict/sample/${encodeURIComponent(className)}?top_k=${topK}&gradcam=${gradcam}`, {
    method: 'POST',
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || 'Sample prediction failed.');
  }
  return data;
}

export async function getMetrics() {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error(`Metrics request failed: ${res.statusText}`);
  return await res.json();
}

export async function getSampleImages() {
  const res = await fetch(`${API_BASE}/samples`);
  if (!res.ok) throw new Error(`Sample images request failed: ${res.statusText}`);
  return await res.json();
}
