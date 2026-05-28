import { Brain, Gauge, Layers, Network, Radar, Route, Shuffle, SplitSquareHorizontal } from "lucide-react";
import type { ReactNode } from "react";

export type Strategy = "random" | "margin" | "entropy" | "bald" | "badge" | "dqn" | "double_dqn" | "dueling_dqn";
export type StrategyKind = "classical" | "rl";

export interface StrategyDefinition {
  id: Strategy;
  notebookName: string;
  label: string;
  shortLabel: string;
  kind: StrategyKind;
  icon: ReactNode;
  desc: string;
  classifierStrategy?: string;
  checkpointHints?: string[];
  qNetHints?: string[];
}

export const strategyDefinitions: StrategyDefinition[] = [
  {
    id: "random",
    notebookName: "random",
    label: "Random",
    shortLabel: "Random",
    kind: "classical",
    icon: <Shuffle className="h-4 w-4" />,
    desc: "Baseline that samples without model uncertainty.",
    classifierStrategy: "random",
  },
  {
    id: "margin",
    notebookName: "margin",
    label: "Margin",
    shortLabel: "Margin",
    kind: "classical",
    icon: <SplitSquareHorizontal className="h-4 w-4" />,
    desc: "Queries samples where the top two class probabilities are closest.",
    classifierStrategy: "margin",
  },
  {
    id: "entropy",
    notebookName: "entropy",
    label: "Entropy",
    shortLabel: "Entropy",
    kind: "classical",
    icon: <Gauge className="h-4 w-4" />,
    desc: "Queries samples with the highest predictive uncertainty.",
    classifierStrategy: "entropy",
  },
  {
    id: "bald",
    notebookName: "bald",
    label: "BALD",
    shortLabel: "BALD",
    kind: "classical",
    icon: <Radar className="h-4 w-4" />,
    desc: "Bayesian active learning baseline from the notebook export.",
    classifierStrategy: "bald",
  },
  {
    id: "badge",
    notebookName: "badge",
    label: "BADGE",
    shortLabel: "BADGE",
    kind: "classical",
    icon: <Layers className="h-4 w-4" />,
    desc: "Gradient-embedding active learning baseline from the notebook export.",
    classifierStrategy: "badge",
  },
  {
    id: "dqn",
    notebookName: "DQN",
    label: "DQN",
    shortLabel: "DQN",
    kind: "rl",
    icon: <Brain className="h-4 w-4" />,
    desc: "RL policy that learns when to predict or request a label.",
    classifierStrategy: "entropy",
    checkpointHints: ["dqn_sota.pt", "dqn_agent.pt", "dqn_vanilla.pt"],
    qNetHints: ["_dqn_q_net.pt"],
  },
  {
    id: "double_dqn",
    notebookName: "Double DQN",
    label: "Double DQN",
    shortLabel: "Double",
    kind: "rl",
    icon: <Route className="h-4 w-4" />,
    desc: "RL policy with decoupled action selection and value estimation.",
    classifierStrategy: "entropy",
    checkpointHints: ["double_dqn_sota.pt", "dqn_double.pt"],
    qNetHints: ["_double_dqn_q_net.pt"],
  },
  {
    id: "dueling_dqn",
    notebookName: "Dueling DQN",
    label: "Dueling DQN",
    shortLabel: "Dueling",
    kind: "rl",
    icon: <Network className="h-4 w-4" />,
    desc: "RL policy with separate value and advantage heads.",
    classifierStrategy: "entropy",
    checkpointHints: ["dueling_dqn_sota.pt", "dqn_dueling.pt"],
    qNetHints: ["_dueling_dqn_q_net.pt"],
  },
];

export const strategyById = Object.fromEntries(
  strategyDefinitions.map((definition) => [definition.id, definition]),
) as Record<Strategy, StrategyDefinition>;

export function normalizeStrategyName(name: string) {
  return name.toLowerCase().replace(/[\s()-]+/g, "_").replace(/^rl_/, "").replace(/_+$/g, "");
}

export function strategyForNotebookName(name: string) {
  const normalized = normalizeStrategyName(name);
  if (normalized === "double_dqn") return "double_dqn";
  if (normalized === "dueling_dqn") return "dueling_dqn";
  if (normalized === "dqn") return "dqn";
  return strategyDefinitions.find((strategy) => strategy.notebookName.toLowerCase() === name.toLowerCase())?.id ?? normalized;
}

export function isRLStrategy(strategy: Strategy) {
  return strategyById[strategy].kind === "rl";
}
