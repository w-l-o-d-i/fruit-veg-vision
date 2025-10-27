interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface PredictionResult {
  label: string;
  confidence: number;
  weights: {
    min: string;
    avg: string;
    max: string;
  };
}

export interface HealthCheck {
  status: string;
  model_loaded: boolean;
  labels_loaded: boolean;
  weights_loaded: boolean;
}

const API_BASE_URL = '/api';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || 'Something went wrong');
  }
  return response.json();
}

export const api = {
  // Health check
  async checkHealth(): Promise<HealthCheck> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return handleResponse<HealthCheck>(response);
  },

  // Predict endpoint
  async predict(imageFile: File): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    const result = await handleResponse<{
      success: boolean;
      prediction?: PredictionResult;
      error?: string;
    }>(response);

    if (!result.success || !result.prediction) {
      throw new Error(result.error || 'Prediction failed');
    }

    return result.prediction;
  },

  // Add more API endpoints here as needed
};

export default api;
