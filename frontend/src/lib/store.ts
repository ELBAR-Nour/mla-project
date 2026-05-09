import { create } from "zustand";

export type Strategy = "rl" | "entropy" | "random";
export type Dataset = "PneumoniaMNIST" | "BreastMNIST";
export type Action = "label" | "predict";

export interface Sample {
  id: string;
  dataset: Dataset;
  trueLabel: 0 | 1;
  probs: [number, number];
  entropy: number;
  confidence: number;
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
  accuracy: number;
  policyConfidence: number;
  expectedReward: number;
  exploration: boolean;
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
  stepOnce: () => void;
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

function buildPool(dataset: Dataset, n = 80): Sample[] {
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
      dataset,
      trueLabel,
      probs,
      entropy: +entropy.toFixed(3),
      confidence: +confidence.toFixed(3),
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
  if (strategy === "random") {
    const action: Action = rng() > 0.5 ? "label" : "predict";
    return { action, expectedReward: 0, policyConfidence: 0.5, exploration: true };
  }
  if (strategy === "entropy") {
    const action: Action = s.entropy > 0.5 ? "label" : "predict";
    return {
      action,
      expectedReward: s.entropy - 0.5,
      policyConfidence: Math.abs(s.entropy - 0.5) + 0.5,
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
  const pred = s.probs[1] > 0.5 ? 1 : 0;
  const correct = pred === s.trueLabel;
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
  strategy: "rl",
  budget: 60,
  rl: { epsilon: 0.1, rewardScale: 1.0, batchSize: 16, learningRate: 0.0003, discount: 0.95 },
  theme: "light",
  visibleStrategies: { rl: true, entropy: true, random: true },

  samples: buildPool(initialDataset),
  currentIndex: 0,
  history: [],
  running: false,
  speedMs: 600,
  experimentStarted: false,

  setDataset: (d) => set({ dataset: d, samples: buildPool(d), currentIndex: 0, history: [] }),
  setStrategy: (s) => set({ strategy: s }),
  setBudget: (b) => set({ budget: b }),
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
      samples: buildPool(st.dataset),
    })),

  resetExperiment: () =>
    set((st) => ({
      experimentStarted: false,
      currentIndex: 0,
      history: [],
      running: false,
      samples: buildPool(st.dataset),
    })),

  setRunning: (r) => set({ running: r }),

  stepOnce: () => {
    const st = get();
    const idx = st.currentIndex;
    if (idx >= st.samples.length) {
      set({ running: false });
      return;
    }
    const used = st.history.filter((h) => h.action === "label").length;
    if (used >= st.budget && st.strategy !== "random") {
      // budget exhausted → force predict
    }
    const sample = st.samples[idx];
    const rng = mulberry32(idx * 7 + (st.strategy === "rl" ? 1 : st.strategy === "entropy" ? 2 : 3));
    const decision = pickAction(sample, st.strategy, st.rl, idx, rng);
    let action = decision.action;
    if (action === "label" && used >= st.budget) action = "predict";
    const { reward, correct } = computeReward(sample, action, st.rl);

    const correctSoFar =
      st.history.reduce((a, h) => a + (h.correct ? 1 : 0), 0) + (correct ? 1 : 0);
    const accuracy = correctSoFar / (st.history.length + 1);
    // AUC saturates with labels acquired
    const labelsCount = used + (action === "label" ? 1 : 0);
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
      accuracy: +accuracy.toFixed(3),
      policyConfidence: decision.policyConfidence,
      expectedReward: decision.expectedReward,
      exploration: decision.exploration,
    };
    set({ history: [...st.history, record], currentIndex: idx + 1 });
  },

  budgetUsed: () => get().history.filter((h) => h.action === "label").length,
  remaining: () => get().budget - get().history.filter((h) => h.action === "label").length,
}));
