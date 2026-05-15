import axios, { AxiosInstance } from 'axios';

export interface DatasetInfo {
  name: string;
  split: string;
  total_samples: number;
  test_samples: number;
  image_shape: number[];
  n_classes: number;
  label_name: string;
  labeled_count: number;
  unlabeled_count: number;
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

export interface StrategyResult {
  strategy: string;
  budget: number;
  queries: number[];
  auc_scores: number[];
  total_labeled: number;
  final_auc: number;
}

export interface StrategyComparison {
  strategies: string[];
  budget: number;
  runs_per_strategy: number;
  comparison: Record<string, {
    runs: number;
    avg_final_auc: number;
    std_final_auc: number;
    avg_queries: number;
    auc_trajectory: number[];
  }>;
}

export class MLService {
  private client: AxiosInstance;
  private baseUrl: string;

  constructor(baseUrl: string = process.env.ML_SERVICE_URL || 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.client = axios.create({
      baseURL: baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('ML Service Error:', error.response?.data || error.message);
        throw error;
      },
    );
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  async isHealthy(): Promise<boolean> {
    try {
      const response = await this.client.get('/health');
      return response.data.status === 'healthy';
    } catch (error) {
      return false;
    }
  }

  // Dataset endpoints
  async initializeDataset(datasetName: string = 'PneumoniaMNIST', split: string = 'train', sampleSize?: number): Promise<any> {
    const response = await this.client.post('/dataset/initialize', {
      dataset_name: datasetName,
      split,
      sample_size: sampleSize,
    });
    return response.data;
  }

  async getDatasetInfo(): Promise<DatasetInfo> {
    const response = await this.client.get('/dataset/info');
    return response.data;
  }

  async getSample(imageId: number): Promise<any> {
    const response = await this.client.get(`/dataset/sample/${imageId}`);
    return response.data;
  }

  async getBatch(count: number = 10): Promise<any[]> {
    const response = await this.client.get(`/dataset/batch?count=${count}`);
    return response.data.samples;
  }

  // Annotation endpoints
  async processAnnotation(imageId: number, action: string, budgetRemaining: number): Promise<AnnotationResult> {
    const response = await this.client.post('/annotate', {
      image_id: imageId,
      action,
      budget_remaining: budgetRemaining,
    });
    return response.data;
  }

  // Strategy endpoints
  async runStrategy(strategy: string, queries: number): Promise<StrategyResult> {
    const response = await this.client.get(`/strategy/${strategy}?queries=${queries}`);
    return response.data;
  }

  async compareStrategies(queryBudget: number, numRuns: number, strategies: string[]): Promise<StrategyComparison> {
    const response = await this.client.post('/strategy/compare', {
      query_budget: queryBudget,
      num_runs: numRuns,
      strategies,
    });
    return response.data;
  }

  // Model endpoints
  async getModelMetrics(): Promise<any> {
    const response = await this.client.get('/model/metrics');
    return response.data;
  }

  async getConfusionMatrix(): Promise<any> {
    const response = await this.client.get('/model/confusion-matrix');
    return response.data;
  }

  async getROCCurve(): Promise<any> {
    const response = await this.client.get('/model/roc');
    return response.data;
  }

  // RL Agent endpoints
  async getRLState(): Promise<any> {
    const response = await this.client.get('/rl/state');
    return response.data;
  }

  async getRLPolicy(): Promise<any> {
    const response = await this.client.get('/rl/policy');
    return response.data;
  }

  async trainRLAgent(episodes: number, learningRate: number = 0.001, gamma: number = 0.99): Promise<any> {
    const response = await this.client.post('/rl/train-step', {
      episodes,
      learning_rate: learningRate,
      gamma,
    });
    return response.data;
  }

  async resetRLAgent(): Promise<any> {
    const response = await this.client.post('/rl/reset');
    return response.data;
  }

  // Checkpoint endpoints
  async loadCheckpoint(checkpointName: string): Promise<any> {
    const response = await this.client.post('/checkpoints/load', {
      checkpoint_name: checkpointName,
    });
    return response.data;
  }

  async runInference(inputData: any): Promise<any> {
    const response = await this.client.post('/inference/run', {
      input_data: inputData,
    });
    return response.data;
  }
}