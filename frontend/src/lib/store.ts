import { create } from "zustand";
import {
  getDatasetBatch,
  getModelArtifacts,
  initializeDataset,
  processAnnotation,
  runModelInference,
  runRLDecision,
  type DatasetSample,
  type ModelArtifact,
  type RLDecisionResult,
} from "./api";
import { DEFAULT_ANNOTATION_BUDGET, MAX_ANNOTATION_BUDGET } from "./constants";
import { isRLStrategy, strategyById, type Strategy } from "./strategies";

export type Dataset = "PneumoniaMNIST" | "BreastMNIST";
export type Action = "label" | "predict";

export interface Sample {
  id: string;
  imageId: number;
  dataset: Dataset;
  imageBase64?: string;
  trueLabel: number | null;
  predictedLabel?: number;
  classLabels?: string[];
  probs: [number, number];
  entropy: number;
  confidence: number;
  acquisitionScore?: number;
  acquisitionLabel?: string;
  recommendedAction?: Action;
  policyConfidence?: number;
  expectedReward?: number;
  liveLoaded?: boolean;
}

export interface StepRecord {
  step: number;
  sampleId: string;
  action: Action;
  reward: number;
  correct: boolean;
  entropy: number;
  confidence: number;
  auc: number;
  policyConfidence: number;
  expectedReward: number;
  exploration: boolean;
  isPolicyReward: boolean;
  budgetUsed: number;
  budgetRemaining: number;
}

export interface RLConfig {
  epsilon: number;
  rewardScale: number;
  batchSize: number;
  learningRate: number;
  discount: number;
}

interface AppState {
  // Config
  dataset: Dataset;
  strategy: Strategy;
  budget: number;
  rl: RLConfig;
  theme: "light" | "dark";
  visibleStrategies: Record<Strategy, boolean>;

  // Simulation
  samples: Sample[];
  currentIndex: number;
  history: StepRecord[];
  running: boolean;
  speedMs: number;
  experimentStarted: boolean;
  dataMode: "live" | "synthetic";
  loadingSamples: boolean;
  loadingDecision: boolean;
  liveError: string | null;
  modelName: string;
  checkpointName: string;

  // Setters
  setDataset: (d: Dataset) => void;
  setStrategy: (s: Strategy) => void;
  setBudget: (b: number) => void;
  setRL: (k: keyof RLConfig, v: number) => void;
  setSpeed: (n: number) => void;
  toggleTheme: () => void;
  toggleStrategy: (s: Strategy) => void;

  // Engine
  startExperiment: () => void;
  resetExperiment: () => void;
  loadLiveSamples: () => Promise<void>;
  loadSampleDecision: (index: number) => Promise<void>;
  stepOnce: () => Promise<void>;
  setRunning: (r: boolean) => void;

  // Derived helpers
  budgetUsed: () => number;
  remaining: () => number;
}

// Deterministic PRNG so curves are stable
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const livePoolSize = MAX_ANNOTATION_BUDGET;

function sampleDisplayId(dataset: Dataset, imageId: number) {
  const prefix = dataset === "PneumoniaMNIST" ? "PNEU" : "BRST";
  return `${prefix}-${String(imageId).padStart(5, "0")}`;
}

function toBinaryProbTuple(probs: number[]): [number, number] {
  if (probs.length >= 2) return [probs[0] ?? 0, probs[1] ?? 0];
  if (probs.length === 1) return [1 - probs[0], probs[0]];
  return [0.5, 0.5];
}

function actionFromService(action?: string): Action | undefined {
  if (!action) return undefined;
  return action === "request_label" || action === "label" ? "label" : "predict";
}

function pickClassifier(artifacts: ModelArtifact[], strategy: Strategy = "entropy") {
  const classifierStrategy = strategyById[strategy].classifierStrategy;
  const matching = classifierStrategy
    ? artifacts.find((artifact) => artifact.strategy === classifierStrategy && !artifact.error)
    : null;
  return (
    matching ??
    artifacts.find((artifact) => artifact.strategy === "entropy" && !artifact.error) ??
    artifacts.find((artifact) => !artifact.error)
  );
}

function pickCheckpoint(artifacts: ModelArtifact[], strategy: Strategy = "dueling_dqn") {
  const definition = strategyById[strategy];
  const hints = [...(definition.qNetHints ?? []), ...(definition.checkpointHints ?? [])];
  const matching = hints
    .map((hint) => artifacts.find((artifact) => artifact.name.includes(hint) && !artifact.error))
    .find(Boolean);
  return (
    matching ??
    artifacts.find((artifact) => artifact.name.includes("dueling") && !artifact.error) ??
    artifacts.find((artifact) => !artifact.error)
  );
}

function acquisitionForStrategy(strategy: Strategy, prediction: { entropy: number; confidence: number; margin: number }) {
  if (strategy === "random") {
    return { score: undefined, label: "random draw" };
  }
  if (strategy === "margin") {
    return { score: 1 - prediction.margin, label: "1 - margin" };
  }
  if (strategy === "entropy" || strategy === "bald") {
    return { score: prediction.entropy, label: strategy === "bald" ? "BALD proxy entropy" : "entropy" };
  }
  if (strategy === "badge") {
    return { score: 1 - prediction.confidence, label: "1 - confidence" };
  }
  return { score: undefined, label: "policy q-value" };
}

function sampleFromDataset(dataset: Dataset, sample: DatasetSample): Sample {
  return {
    id: sampleDisplayId(dataset, sample.image_id),
    imageId: sample.image_id,
    dataset,
    imageBase64: sample.image_base64,
    trueLabel: null,
    probs: [0.5, 0.5],
    entropy: 0,
    confidence: 0,
    liveLoaded: false,
  };
}

function errorMessage(exc: unknown, fallback: string) {
  return exc instanceof Error ? exc.message : fallback;
}

function buildPool(dataset: Dataset, n = MAX_ANNOTATION_BUDGET): Sample[] {
  const rng = mulberry32(dataset === "PneumoniaMNIST" ? 42 : 91);
  const prefix = dataset === "PneumoniaMNIST" ? "PNEU" : "BRST";
  return Array.from({ length: n }, (_, i) => {
    const trueLabel = (rng() > 0.45 ? 1 : 0) as 0 | 1;
    // Simulate a noisy classifier: skew probability toward true label
    const skew = 0.45 + rng() * 0.5;
    const pTrue = trueLabel === 1 ? skew : 1 - skew;
    const p1 = +pTrue.toFixed(3);
    const p0 = +(1 - p1).toFixed(3);
    const probs: [number, number] = [p0, p1];
    const entropy = -probs.reduce(
      (a, p) => a + (p > 0 ? p * Math.log2(p) : 0),
      0,
    );
    const confidence = Math.max(p0, p1);
    return {
      id: `${prefix}-${String(10000 + i).slice(1)}`,
      imageId: i,
      dataset,
      trueLabel,
      probs,
      entropy: +entropy.toFixed(3),
      confidence: +confidence.toFixed(3),
      liveLoaded: true,
    };
  });
}

function pickAction(
  s: Sample,
  strategy: Strategy,
  rl: RLConfig,
  step: number,
  rng: () => number,
): { action: Action; expectedReward: number; policyConfidence: number; exploration: boolean } {
  if (!isRLStrategy(strategy)) {
    return {
      action: "label",
      expectedReward: s.acquisitionScore ?? 0,
      policyConfidence: s.confidence || 0.5,
      exploration: strategy === "random",
    };
  }

  if (s.recommendedAction) {
    return {
      action: s.recommendedAction,
      expectedReward: s.expectedReward ?? 0,
      policyConfidence: s.policyConfidence ?? 0.5,
      exploration: false,
    };
  }

  // RL: ε-greedy over a learned-ish utility = entropy + α·step decay
  const exploration = rng() < rl.epsilon;
  const learned = 0.6 + Math.min(0.35, step / 250);
  const utility = s.entropy * learned + (1 - s.confidence) * 0.5;
  const action: Action = exploration
    ? rng() > 0.5
      ? "label"
      : "predict"
    : utility > 0.55
      ? "label"
      : "predict";
  return {
    action,
    expectedReward: +(utility * rl.rewardScale).toFixed(3),
    policyConfidence: +Math.min(0.99, 0.55 + utility * 0.4).toFixed(3),
    exploration,
  };
}

function computeReward(s: Sample, action: Action, rl: RLConfig): { reward: number; correct: boolean } {
  const pred = s.predictedLabel ?? (s.probs[1] > 0.5 ? 1 : 0);
  const correct = s.trueLabel === null ? true : pred === s.trueLabel;
  if (action === "label") {
    // Cost of querying expert; payoff scales with information gained
    const reward = (s.entropy - 0.4) * rl.rewardScale - 0.15;
    return { reward: +reward.toFixed(3), correct: true };
  }
  // Predict: gain if correct, penalty if wrong
  const reward = correct ? 0.2 : -s.entropy * rl.rewardScale;
  return { reward: +reward.toFixed(3), correct };
}

const initialDataset: Dataset = "PneumoniaMNIST";

export const useApp = create<AppState>((set, get) => ({
  dataset: initialDataset,
  strategy: "dueling_dqn",
  budget: DEFAULT_ANNOTATION_BUDGET,
  rl: { epsilon: 0.1, rewardScale: 1.0, batchSize: 16, learningRate: 0.0003, discount: 0.95 },
  theme: "light",
  visibleStrategies: {
    random: true,
    margin: true,
    entropy: true,
    bald: true,
    badge: true,
    dqn: true,
    double_dqn: true,
    dueling_dqn: true,
  },

  samples: buildPool(initialDataset),
  currentIndex: 0,
  history: [],
  running: false,
  speedMs: 600,
  experimentStarted: false,
  dataMode: "synthetic",
  loadingSamples: false,
  loadingDecision: false,
  liveError: null,
  modelName: "",
  checkpointName: "",

  setDataset: (d) => set({
    dataset: d,
    samples: buildPool(d),
    currentIndex: 0,
    history: [],
    dataMode: "synthetic",
    liveError: null,
  }),
  setStrategy: (s) => set({ strategy: s }),
  setBudget: (b) => set({ budget: Math.min(Math.max(Math.round(b), 1), MAX_ANNOTATION_BUDGET) }),
  setRL: (k, v) => set((st) => ({ rl: { ...st.rl, [k]: v } })),
  setSpeed: (n) => set({ speedMs: n }),
  toggleTheme: () =>
    set((s) => {
      const next = s.theme === "light" ? "dark" : "light";
      if (typeof document !== "undefined") {
        document.documentElement.classList.toggle("dark", next === "dark");
      }
      return { theme: next };
    }),
  toggleStrategy: (s) =>
    set((st) => ({
      visibleStrategies: { ...st.visibleStrategies, [s]: !st.visibleStrategies[s] },
    })),

  startExperiment: () =>
    set((st) => ({
      experimentStarted: true,
      currentIndex: 0,
      history: [],
      running: false,
      samples: [],
      dataMode: "live",
      loadingSamples: true,
      loadingDecision: false,
      liveError: null,
      modelName: "",
      checkpointName: "",
    })),

  resetExperiment: () =>
    set((st) => ({
      experimentStarted: false,
      currentIndex: 0,
      history: [],
      running: false,
      samples: buildPool(st.dataset),
      dataMode: "synthetic",
      loadingSamples: false,
      loadingDecision: false,
      liveError: null,
    })),

  setRunning: (r) => set({ running: r }),

  loadLiveSamples: async () => {
    const st = get();
    set({ loadingSamples: true, liveError: null, dataMode: "live" });
    try {
      const [artifacts] = await Promise.all([
        getModelArtifacts(),
        initializeDataset({ dataset_name: st.dataset, split: "train" }),
      ]);
      const batch = await getDatasetBatch(livePoolSize);
      const modelName = pickClassifier(artifacts.classifiers, st.strategy)?.name ?? artifacts.active_model ?? "";
      const checkpointName = isRLStrategy(st.strategy)
        ? pickCheckpoint(artifacts.checkpoints, st.strategy)?.name ?? artifacts.active_checkpoint ?? ""
        : "";
      set({
        samples: batch.map((sample) => sampleFromDataset(st.dataset, sample)),
        currentIndex: 0,
        history: [],
        loadingSamples: false,
        modelName,
        checkpointName,
        liveError: modelName ? null : "No classifier model artifact is available for live inference.",
      });
    } catch (exc) {
      set({
        samples: [],
        running: false,
        loadingSamples: false,
        liveError: errorMessage(exc, "Could not load dataset samples from the ML service."),
      });
    }
  },

  loadSampleDecision: async (index) => {
    const st = get();
    const sample = st.samples[index];
    if (!sample || sample.liveLoaded || st.loadingDecision || !st.modelName) return;

    set({ loadingDecision: true, liveError: null });
    try {
      const prediction = await runModelInference({
        model_name: st.modelName,
        image_id: sample.imageId,
        dataset_name: st.dataset,
        split: "train",
      });

      let policy: RLDecisionResult | null = null;
      let policyError: string | null = null;
      if (isRLStrategy(st.strategy) && st.checkpointName) {
        try {
          policy = await runRLDecision({
            checkpoint_name: st.checkpointName,
            model_name: st.modelName,
            image_id: sample.imageId,
            dataset_name: st.dataset,
            split: "train",
            budget_remaining: st.remaining(),
            max_budget: st.budget,
          });
        } catch (exc) {
          policyError = errorMessage(exc, "RL decision failed.");
        }
      }

      set((currentState) => {
        const current = currentState.samples[index];
        if (!current || current.imageId !== sample.imageId) {
          return { loadingDecision: false, liveError: policyError };
        }

        const samples = [...currentState.samples];
        const recommendedAction = actionFromService(policy?.recommended_action);
        const acquisition = acquisitionForStrategy(currentState.strategy, prediction);
        samples[index] = {
          ...current,
          imageBase64: prediction.image_base64 ?? policy?.image_base64 ?? current.imageBase64,
          trueLabel: prediction.true_label,
          predictedLabel: prediction.predicted_label,
          classLabels: prediction.class_labels,
          probs: toBinaryProbTuple(prediction.probabilities),
          entropy: +prediction.entropy.toFixed(3),
          confidence: +prediction.confidence.toFixed(3),
          acquisitionScore: acquisition.score,
          acquisitionLabel: acquisition.label,
          recommendedAction,
          policyConfidence: policy?.policy_confidence,
          expectedReward:
            policy && policy.recommended_action_index < policy.q_values.length
              ? policy.q_values[policy.recommended_action_index]
              : undefined,
          liveLoaded: true,
        };
        return { samples, loadingDecision: false, liveError: policyError };
      });
    } catch (exc) {
      set({
        loadingDecision: false,
        liveError: errorMessage(exc, "Could not run model inference for the selected sample."),
      });
    }
  },

  stepOnce: async () => {
    let st = get();
    const idx = st.currentIndex;
    if (idx >= st.samples.length) {
      set({ running: false });
      return;
    }
    if (st.dataMode === "live" && !st.samples[idx]?.liveLoaded) {
      await get().loadSampleDecision(idx);
      st = get();
      if (!st.samples[idx]?.liveLoaded) return;
    }
    const used = st.budgetUsed();
    if (used >= st.budget && st.strategy !== "random") {
      // budget exhausted → force predict
    }
    const sample = st.samples[idx];
    const rng = mulberry32(idx * 7 + (isRLStrategy(st.strategy) ? 1 : st.strategy === "entropy" ? 2 : 3));
    const decision = pickAction(sample, st.strategy, st.rl, idx, rng);
    let action = decision.action;
    if (action === "label" && used >= st.budget) action = "predict";
    const outcome = computeReward(sample, action, st.rl);
    let reward = isRLStrategy(st.strategy) ? outcome.reward : 0;
    let correct = outcome.correct;
    let budgetUsed = used + (action === "label" ? 1 : 0);
    let budgetRemaining = Math.max(st.budget - budgetUsed, 0);

    if (st.dataMode === "live") {
      try {
        const annotation = await processAnnotation({
          image_id: sample.imageId,
          action: action === "label" ? "request_label" : "predict",
          budget_remaining: Math.max(st.budget - used, 0),
          max_budget: st.budget,
        });
        budgetUsed = annotation.budget_used;
        budgetRemaining = annotation.budget_remaining;
        correct = annotation.correct;
        if (isRLStrategy(st.strategy)) reward = +annotation.reward.toFixed(3);
      } catch (exc) {
        set({
          liveError: errorMessage(exc, "Could not sync the annotation budget with the ML service."),
        });
      }
    }

    // AUC saturates with labels acquired
    const labelsCount = budgetUsed;
    const auc = +Math.min(0.97, 0.62 + 0.34 * (1 - Math.exp(-labelsCount / 18))).toFixed(3);

    const record: StepRecord = {
      step: st.history.length + 1,
      sampleId: sample.id,
      action,
      reward,
      correct,
      entropy: sample.entropy,
      confidence: sample.confidence,
      auc,
      policyConfidence: decision.policyConfidence,
      expectedReward: decision.expectedReward,
      exploration: decision.exploration,
      isPolicyReward: isRLStrategy(st.strategy),
      budgetUsed,
      budgetRemaining,
    };
    set({ history: [...st.history, record], currentIndex: idx + 1 });
  },

  budgetUsed: () => get().history.at(-1)?.budgetUsed ?? get().history.filter((h) => h.action === "label").length,
  remaining: () => get().history.at(-1)?.budgetRemaining ?? get().budget - get().history.filter((h) => h.action === "label").length,
}));
