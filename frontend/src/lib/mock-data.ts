// Mock data for the RL Active Learning demo.

export const learningCurve = Array.from({ length: 21 }, (_, i) => {
  const q = i * 25;
  // Smooth saturating curves
  const random = 0.55 + 0.3 * (1 - Math.exp(-q / 250));
  const entropy = 0.6 + 0.32 * (1 - Math.exp(-q / 180));
  const rl = 0.62 + 0.34 * (1 - Math.exp(-q / 120));
  return {
    queries: q,
    random: +random.toFixed(3),
    entropy: +entropy.toFixed(3),
    rl: +rl.toFixed(3),
  };
});

export const rocCurve = Array.from({ length: 21 }, (_, i) => {
  const fpr = i / 20;
  return {
    fpr: +fpr.toFixed(2),
    rl: +Math.min(1, Math.pow(fpr, 0.25)).toFixed(3),
    entropy: +Math.min(1, Math.pow(fpr, 0.35)).toFixed(3),
    random: +Math.min(1, Math.pow(fpr, 0.6)).toFixed(3),
  };
});

export const trainingCurves = Array.from({ length: 30 }, (_, i) => ({
  epoch: i + 1,
  loss: +(1.4 * Math.exp(-i / 8) + 0.12 + Math.random() * 0.04).toFixed(3),
  accuracy: +(0.55 + 0.4 * (1 - Math.exp(-i / 6)) + Math.random() * 0.01).toFixed(3),
}));

export const entropyHistogram = Array.from({ length: 12 }, (_, i) => ({
  bin: +((i + 0.5) * (1 / 12)).toFixed(2),
  count: Math.round(40 * Math.exp(-Math.pow((i - 6) / 3, 2)) + Math.random() * 8),
  selected: Math.round(20 * Math.exp(-Math.pow((i - 9) / 2, 2))),
}));

export const confusionMatrix = [
  [142, 8],
  [11, 119],
];

export const strategyTable = [
  { strategy: "Random Sampling", queries: 320, accuracy: 0.842, auc: 0.871, efficiency: 0.272 },
  { strategy: "Entropy Sampling", queries: 240, accuracy: 0.881, auc: 0.912, efficiency: 0.380 },
  { strategy: "RL Agent", queries: 184, accuracy: 0.913, auc: 0.946, efficiency: 0.514 },
];

export const samplePrediction = {
  imageId: "PNEU-00471",
  classes: ["Normal", "Pneumonia"],
  probs: [0.31, 0.69],
  entropy: 0.62,
  confidence: 0.69,
  recommended: "request" as "request" | "predict",
  expectedReward: 0.41,
  rlConfidence: 0.78,
};
