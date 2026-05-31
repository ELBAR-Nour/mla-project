import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Brain, Compass, TrendingUp } from "lucide-react";
import { ChartContainer } from "@/components/chart-container";
import { MetricCard } from "@/components/metric-card";
import { useApp } from "@/lib/store";
import { isRLStrategy, strategyById } from "@/lib/strategies";

export const Route = createFileRoute("/learning-evolution")({
  head: () => ({
    meta: [
      { title: "Learning Evolution — MedAL" },
      { name: "description", content: "Watch strategy-specific AUC, query usage, and policy metrics evolve over time." },
    ],
  }),
  component: LearningEvolution,
});

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: "12px",
  fontSize: "12px",
};

function LearningEvolution() {
  const { history, strategy } = useApp();

  if (history.length === 0) {
    return <EmptyEvolution />;
  }

  const labels = history.filter((h) => h.action === "label").length;
  const predicts = history.length - labels;
  const explore = history.filter((h) => h.exploration).length;
  const exploit = history.length - explore;
  const totalReward = history.reduce((a, h) => a + h.reward, 0);
  const last = history[history.length - 1];
  const policyMode = isRLStrategy(strategy);
  const strategyLabel = strategyById[strategy].label;

  const policyData = [
    { name: "Request Label", value: labels, color: "var(--color-chart-1)" },
    { name: "Predict", value: predicts, color: "var(--color-chart-4)" },
  ];
  const explorationData = [
    { name: "Exploit", value: exploit, color: "var(--color-chart-1)" },
    { name: "Explore", value: explore, color: "var(--color-chart-2)" },
  ];
  const confidenceData = [
    { name: "High confidence", value: history.filter((h) => h.confidence >= 0.8).length, color: "var(--color-chart-1)" },
    { name: "Medium confidence", value: history.filter((h) => h.confidence >= 0.6 && h.confidence < 0.8).length, color: "var(--color-chart-2)" },
    { name: "Low confidence", value: history.filter((h) => h.confidence < 0.6).length, color: "var(--color-chart-5)" },
  ].filter((item) => item.value > 0);

  // Cumulative reward
  let acc = 0;
  const rewardCurve = history.map((h) => {
    acc += h.reward;
    return { step: h.step, reward: +acc.toFixed(3) };
  });
  const scoreCurve = history.map((h) => ({
    step: h.step,
    score: +h.expectedReward.toFixed(3),
    confidence: h.confidence,
  }));

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <div>
        <h1 className="font-display text-2xl font-semibold md:text-3xl">
          Learning <span className="text-gradient">Evolution</span>
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          {strategyLabel} progress, label usage, and model quality across the simulation.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <MetricCard label="AUC" value={last.auc.toFixed(3)} hint="latest" icon={<TrendingUp className="h-5 w-5" />} />
        {policyMode ? (
          <>
            <MetricCard label="Cumulative Reward" value={totalReward.toFixed(2)} hint="RL reward signal" tone={totalReward >= 0 ? "success" : "danger"} icon={<Brain className="h-5 w-5" />} />
            <MetricCard label="Exploration Rate" value={`${((explore / history.length) * 100).toFixed(0)}%`} hint={`${explore} explore / ${exploit} exploit`} tone="warning" icon={<Compass className="h-5 w-5" />} />
          </>
        ) : (
          <>
            <MetricCard label="Labels Queried" value={`${labels}`} hint={`${strategyLabel} acquisition`} tone="warning" icon={<Brain className="h-5 w-5" />} />
            <MetricCard label="Predictions" value={`${predicts}`} hint="automatic predictions" icon={<Compass className="h-5 w-5" />} />
          </>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <ChartContainer className="lg:col-span-2" title="AUC Over Time" description="Model quality per simulation step">
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={history} margin={{ left: -10, right: 8, top: 8 }}>
              <defs>
                <linearGradient id="evo-auc" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--color-chart-1)" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="var(--color-chart-1)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} domain={[0.4, 1]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Area type="monotone" dataKey="auc" stroke="var(--color-chart-1)" strokeWidth={3} fill="url(#evo-auc)" name="AUC" />
            </AreaChart>
          </ResponsiveContainer>
        </ChartContainer>

        {policyMode ? (
          <ChartContainer title="Cumulative Reward" description="Total reward earned by the agent">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rewardCurve} margin={{ left: -10, right: 8 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={11} />
                <YAxis stroke="var(--muted-foreground)" fontSize={11} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="reward" stroke="var(--color-chart-4)" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        ) : (
          <ChartContainer title="Acquisition Score" description={`${strategyLabel} sample score over time`}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={scoreCurve} margin={{ left: -10, right: 8 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={11} />
                <YAxis stroke="var(--muted-foreground)" fontSize={11} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="score" stroke="var(--color-chart-4)" strokeWidth={3} dot={false} name="Score" />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        )}

        <ChartContainer title="Selection Distribution" description="Label vs predict actions taken">
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={policyData} dataKey="value" innerRadius={50} outerRadius={80} paddingAngle={3}>
                {policyData.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartContainer>

        {policyMode ? (

        <ChartContainer title="Exploration vs Exploitation" description="Epsilon-greedy split during the run">
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={explorationData} dataKey="value" innerRadius={50} outerRadius={80} paddingAngle={3}>
                {explorationData.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartContainer>

        ) : (
          <ChartContainer title="Confidence Bands" description="Classifier confidence of selected samples">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={confidenceData} dataKey="value" innerRadius={50} outerRadius={80} paddingAngle={3}>
                  {confidenceData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </ChartContainer>
        )}

        {policyMode ? (
        <ChartContainer title="Reward Per Step" description="Sparse positive/negative policy signal">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={history} margin={{ left: -10, right: 8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={11} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="reward" stroke="var(--color-chart-5)" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>
        ) : (
          <ChartContainer title="Confidence Per Step" description="Classifier certainty of selected samples">
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={scoreCurve} margin={{ left: -10, right: 8 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="step" stroke="var(--muted-foreground)" fontSize={11} />
                <YAxis stroke="var(--muted-foreground)" fontSize={11} domain={[0, 1]} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="confidence" stroke="var(--color-chart-5)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        )}
      </div>
    </motion.div>
  );
}

function EmptyEvolution() {
  return (
    <div className="grid min-h-[50vh] place-items-center">
      <div className="glass max-w-md rounded-3xl p-8 text-center">
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-gradient-primary text-primary-foreground shadow-elegant">
          <Brain className="h-6 w-6" />
        </div>
        <h2 className="mt-4 font-display text-xl font-semibold">Nothing to learn from yet</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Run a few steps in the Live Simulation. Strategy progress will appear here in real time.
        </p>
      </div>
    </div>
  );
}
