const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:3001/api";

export interface ModelArtifact {
  name: string;
  strategy?: string;
  source?: string;
  architecture?: string;
  n_classes?: number;
  state_dim?: number;
  action_dim?: number;
  size_bytes?: number;
  active?: boolean;
  error?: string;
}

export interface ModelArtifactsResponse {
  classifiers: ModelArtifact[];
  checkpoints: ModelArtifact[];
  active_model: string | null;
  active_checkpoint: string | null;
}

export interface DatasetInitResponse {
  status: string;
  dataset_name: string;
  split: string;
  total_samples: number;
  image_shape: number[];
}

export interface DatasetSample {
  image_id: number;
  image_base64: string;
  shape: number[];
}

export interface InferenceResult {
  status: string;
  dataset_name: string;
  split: string;
  image_id: number | null;
  image_base64: string | null;
  model_name: string;
  predicted_label: number;
  predicted_label_name: string;
  true_label: number | null;
  true_label_name: string | null;
  probabilities: number[];
  class_labels: string[];
  confidence: number;
  entropy: number;
  margin: number;
  correct: boolean | null;
}

export interface RLDecisionResult {
  status: string;
  dataset_name: string;
  split: string;
  image_id: number;
  image_base64: string;
  checkpoint_name: string;
  architecture: string;
  state_dim: number;
  state_features: string[];
  state_vector: number[];
  state_used: number[];
  q_values: number[];
  action_probabilities: number[];
  recommended_action: "predict" | "request_label" | string;
  recommended_action_index: number;
  policy_confidence: number;
  prediction: InferenceResult;
}

export interface AnnotationResult {
  image_id: number;
  action: string;
  reward: number;
  cost: number;
  predicted_label: number;
  true_label: number;
  confidence: number;
  uncertainty: number;
  correct: boolean;
  budget_used: number;
  budget_remaining: number;
}

export interface ConfusionMatrixResponse {
  matrix: number[][];
  labels: Array<string | number>;
  model_name?: string | null;
  dataset_name?: string;
  metrics?: Record<string, number>;
}

export interface ROCResponse {
  fpr: number[];
  tpr: number[];
  thresholds: Array<number | null>;
  model_name?: string | null;
  dataset_name?: string;
}

export interface ExperimentResult {
  strategy: string;
  alc: number;
  final_val_auc: number;
  final_test_auc: number;
  queries: number;
}

export interface ClinicalMetric {
  strategy: string;
  f1: number;
  recall: number;
  precision: number;
  alc: number;
  efficiency: number;
  final_auc: number;
  queries: number;
}

export interface LearningCurvePoint {
  strategy: string;
  step: number;
  queries: number;
  val_auc: number;
  test_auc: number;
  n_labels: number;
}

export interface ExperimentSummary {
  config: Record<string, unknown>;
  artifact_manifest: Record<string, unknown>;
  dataset_summary: Array<Record<string, unknown>>;
  main_results: ExperimentResult[];
  clinical_metrics: ClinicalMetric[];
  learning_curves: LearningCurvePoint[];
  multiseed_summary: Array<Record<string, unknown>>;
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload?.details ??
      payload?.detail ??
      payload?.error ??
      `Request failed with status ${response.status}`;
    throw new Error(String(message));
  }

  return payload as T;
}

export function initializeDataset(input: {
  dataset_name?: string;
  split?: string;
  sample_size?: number;
}) {
  return apiRequest<DatasetInitResponse>("/dataset/initialize", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function getDatasetBatch(count = 10) {
  return apiRequest<DatasetSample[]>(`/dataset/batch?count=${count}`);
}

export function getModelArtifacts() {
  return apiRequest<ModelArtifactsResponse>("/model/artifacts");
}

export function getExperimentSummary() {
  return apiRequest<ExperimentSummary>("/strategy/experiments");
}

export function loadModel(modelName: string) {
  return apiRequest("/model/load", {
    method: "POST",
    body: JSON.stringify({ model_name: modelName }),
  });
}

export function loadCheckpoint(checkpointName: string) {
  return apiRequest("/rl/load-checkpoint", {
    method: "POST",
    body: JSON.stringify({ checkpoint_name: checkpointName }),
  });
}

export function runModelInference(input: {
  model_name?: string;
  image_id?: number;
  image_base64?: string;
  dataset_name?: string;
  split?: string;
}) {
  return apiRequest<InferenceResult>("/model/inference", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

function withQuery(path: string, params?: Record<string, string | undefined>) {
  const search = new URLSearchParams();
  Object.entries(params ?? {}).forEach(([key, value]) => {
    if (value) search.set(key, value);
  });
  const query = search.toString();
  return query ? `${path}?${query}` : path;
}

export function getConfusionMatrix(input?: {
  model_name?: string;
  dataset_name?: string;
}) {
  return apiRequest<ConfusionMatrixResponse>(withQuery("/model/confusion-matrix", input));
}

export function getROCCurve(input?: {
  model_name?: string;
  dataset_name?: string;
}) {
  return apiRequest<ROCResponse>(withQuery("/model/roc", input));
}

export function processAnnotation(input: {
  image_id: number;
  action: "predict" | "request_label";
  budget_remaining: number;
  max_budget?: number;
}) {
  return apiRequest<AnnotationResult>("/annotation", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function runRLDecision(input: {
  checkpoint_name?: string;
  model_name?: string;
  image_id: number;
  dataset_name?: string;
  split?: string;
  budget_remaining?: number;
  max_budget?: number;
}) {
  return apiRequest<RLDecisionResult>("/rl/decision", {
    method: "POST",
    body: JSON.stringify(input),
  });
}
