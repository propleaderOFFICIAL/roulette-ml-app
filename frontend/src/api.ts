const _apiBase =
  import.meta.env.VITE_API_URL ??
  (import.meta.env.PROD ? 'https://roulette-ml-api.onrender.com' : 'http://localhost:8000');
/** URL base senza trailing slash per evitare doppio slash (es. ...com//spins → 404) */
export const API_BASE = typeof _apiBase === 'string' ? _apiBase.replace(/\/$/, '') : _apiBase;

export interface Spin {
  number: number;
  color: string;
  timestamp: string;
}

export interface SpinResponse {
  number: number;
  color: string;
  timestamp: string;
  total_spins: number;
}

export interface PredictionsResponse {
  theoretical: {
    color: Record<string, number>;
    number_probability: number;
  };
  empirical: {
    color: Record<string, number>;
    top_numbers: Array<{ number: number; probability: number }>;
  };
  model: {
    color?: Record<string, number>;
    top_numbers?: Array<{ number: number; probability: number }>;
  } | null;
  betting_areas: {
    dozen: { probabilities: Record<string, number>; prediction: string; confidence: number };
    column: { probabilities: Record<string, number>; prediction: string; confidence: number };
    high_low: { probabilities: Record<string, number>; prediction: string; confidence: number };
    parity: { probabilities: Record<string, number>; prediction: string; confidence: number };
    zero_probability: number;
  } | null;
  total_spins: number;
}

export interface ModelPrediction {
  color?: Record<string, number>;
  top_numbers?: Array<{ number: number; probability: number }>;
}

export interface AdvancedColorPrediction {
  ensemble: Record<string, number>;
  confidence: number;
  models: Record<string, Record<string, number>>;
  agreement: number;
  weights: Record<string, number>;
}

export interface AdvancedNumberPrediction {
  ensemble: Array<{ number: number; probability: number }>;
  confidence: number;
  models: Record<string, Array<{ number: number; probability: number }>>;
  weights: Record<string, number>;
}

export interface ModelInfo {
  trained: boolean;
  total_samples: number;
  models: Record<string, {
    trained: boolean;
    weight: number;
    available: boolean;
  }>;
  min_samples_required: number;
  retrain_interval: number;
}

export interface BettingAreaItem {
  probabilities: Record<string, number>;
  prediction: string;
  confidence: number;
}

export interface BettingAreaPredictions {
  dozen: BettingAreaItem;
  column: BettingAreaItem;
  high_low: BettingAreaItem;
  parity: BettingAreaItem;
  zero_probability: number;
  source: string;
}

export interface AdvancedPredictionsResponse {
  color: AdvancedColorPrediction | null;
  number: AdvancedNumberPrediction | null;
  betting_areas: BettingAreaPredictions | null;
  model_info: ModelInfo;
  total_spins: number;
  error?: string;
}

export interface HotColdNumber {
  number: number;
  deviation: number;
  count: number;
}

export interface SleeperNumber {
  number: number;
  gap: number;
  overdue_by: number;
  urgency: string;
}

export interface StreakInfo {
  length: number;
  value: string | null;
}

export interface PatternAlert {
  type: string;
  value?: string;
  number?: number;
  length?: number;
  severity: string;
  message: string;
}

export interface PatternAnalysis {
  hot_cold: {
    hot: HotColdNumber[];
    cold: HotColdNumber[];
    neutral_count: number;
    total_spins: number;
  };
  sleepers: {
    sleepers: SleeperNumber[];
    max_gap: number;
    total_sleepers: number;
  };
  streaks: {
    current_streaks: Record<string, StreakInfo>;
    max_streaks: Record<string, StreakInfo>;
    pattern_alerts: PatternAlert[];
  };
  sector_bias: {
    bias_detected: boolean;
    sectors: Record<string, {
      expected: number;
      actual: number;
      deviation: number;
      bias_level: string;
    }>;
    alerts: Array<{
      sector: string;
      deviation: number;
      direction: string;
      message: string;
    }>;
  };
  alerts: PatternAlert[];
  alert_count: number;
}

export interface PatternAnalysisResponse {
  patterns: PatternAnalysis;
  total_spins: number;
  error?: string;
}

export interface ChiSquaredResult {
  statistic: number;
  p_value: number;
  significant: boolean;
  interpretation: string;
}

export interface MarkovResult {
  color_transition_matrix: Record<string, Record<string, number>>;
  current_state: {
    last_color: string;
    last_dozen: string;
  };
  next_color_prediction: Record<string, number>;
}

export interface BayesianResult {
  prior: Record<string, number>;
  posterior: Record<string, number>;
  confidence: number;
  predicted_color: string;
}

export interface EntropyResult {
  number_entropy: number;
  color_entropy: number;
  randomness_score: number;
  interpretation: string;
}

export interface MonteCarloResult {
  type: string;
  simulated_probabilities?: Record<string, number>;
  empirical_probabilities?: Record<string, number>;
  theoretical_probabilities?: Record<string, number>;
  top_simulated?: Array<{ number: number; probability: number }>;
  iterations: number;
}

export interface StatisticalAnalysis {
  chi_squared: {
    number: ChiSquaredResult;
    color: ChiSquaredResult;
  };
  monte_carlo: {
    color: MonteCarloResult;
    number: MonteCarloResult;
  };
  markov: MarkovResult;
  bayesian: BayesianResult;
  entropy: EntropyResult;
  runs_test: {
    runs_observed: number;
    runs_expected: number;
    z_score: number;
    p_value: number;
    random: boolean;
    interpretation: string;
  };
  overall_assessment: string;
  bias_indicators: number;
}

export interface StatisticalAnalysisResponse {
  statistics: StatisticalAnalysis;
  total_spins: number;
  error?: string;
}

// Timeout lungo per cold start Render (piano free ~30-60 s)
const API_TIMEOUT_MS = 70000;

async function fetchWithTimeout(url: string, options: RequestInit = {}): Promise<Response> {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), API_TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...options, signal: ctrl.signal });
    clearTimeout(id);
    return res;
  } catch (e) {
    clearTimeout(id);
    throw e;
  }
}

// API Functions

export async function addSpin(number: number): Promise<SpinResponse> {
  const response = await fetchWithTimeout(`${API_BASE}/spins`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ number }),
  });
  if (!response.ok) {
    throw new Error('Failed to add spin');
  }
  return response.json();
}

export async function getSpins(limit?: number): Promise<{ spins: Spin[]; total: number }> {
  const url = limit != null ? `${API_BASE}/spins?limit=${limit}` : `${API_BASE}/spins`;
  const response = await fetchWithTimeout(url);
  if (!response.ok) {
    throw new Error('Failed to fetch spins');
  }
  return response.json();
}

export async function clearSpins(): Promise<{ message: string; cleared_count: number }> {
  const response = await fetchWithTimeout(`${API_BASE}/spins`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to clear spins');
  }
  return response.json();
}

export async function getPredictions(): Promise<PredictionsResponse> {
  const response = await fetchWithTimeout(`${API_BASE}/predictions`);
  if (!response.ok) {
    throw new Error('Failed to fetch predictions');
  }
  return response.json();
}

export async function getAdvancedPredictions(): Promise<AdvancedPredictionsResponse> {
  const response = await fetchWithTimeout(`${API_BASE}/predictions/advanced`);
  if (!response.ok) {
    throw new Error('Failed to fetch advanced predictions');
  }
  return response.json();
}

export async function getPatternAnalysis(): Promise<PatternAnalysisResponse> {
  const response = await fetchWithTimeout(`${API_BASE}/analysis/patterns`);
  if (!response.ok) {
    throw new Error('Failed to fetch pattern analysis');
  }
  return response.json();
}

export async function getStatisticalAnalysis(): Promise<StatisticalAnalysisResponse> {
  const response = await fetchWithTimeout(`${API_BASE}/analysis/statistics`);
  if (!response.ok) {
    throw new Error('Failed to fetch statistical analysis');
  }
  return response.json();
}

export async function getModelInfo(): Promise<{ ensemble: ModelInfo; legacy_predictor: { trained: boolean; window_size: number }; total_spins: number }> {
  const response = await fetchWithTimeout(`${API_BASE}/models/info`);
  if (!response.ok) {
    throw new Error('Failed to fetch model info');
  }
  return response.json();
}
