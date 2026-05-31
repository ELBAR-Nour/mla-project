import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, Coins, Gauge, Play, FlaskConical } from "lucide-react";
import { MetricCard } from "@/components/metric-card";
import { ChartContainer } from "@/components/chart-container";
import { BudgetBar } from "@/components/budget-bar";
import { Button } from "@/components/ui/button";
import { Toggle } from "@/components/ui/toggle";
import { getExperimentSummary, type ExperimentSummary } from "@/lib/api";
import { useApp } from "@/lib/store";
import { strategyDefinitions, strategyForNotebookName } from "@/lib/strategies";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard — MedAL" },
      { name: "description", content: "Experiment overview and learning efficiency for active-learning and RL strategies." },
    ],
  }),
  component: Dashboard,
});

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: "12px",
  fontSize: "12px",
};

function Dashboard() {
  const { visibleStrategies, toggleStrategy, budget, budgetUsed, remaining, history, strategy } = useApp();
  const [summary, setSummary] = useState<ExperimentSummary | null>(null);
  useEffect(() => {
    void getExperimentSummary().then(setSummary).catch(() => setSummary(null));
  }, []);

  const curveData = useMemo(() => {
    if (!summary) return [];
    const rows = new Map<number, Record<string, number>>();
    for (const point of summary.learning_curves) {
      const id = strategyForNotebookName(point.strategy);
      const existing = rows.get(point.queries) ?? { queries: point.queries };
      existing[id] = point.test_auc;
      rows.set(point.queries, existing);
    }
    return [...rows.values()].sort((a, b) => a.queries - b.queries);
  }, [summary]);

  const selectedNotebookName = strategyDefinitions.find((item) => item.id === strategy)?.notebookName ?? strategy;
  const selectedResult = summary?.main_results.find(
    (row) => row.strategy.toLowerCase() === selectedNotebookName.toLowerCase(),
  );
  const used = budgetUsed();
  const budgetRemaining = remaining();
  const last = history[history.length - 1];
  const auc = last ? last.auc.toFixed(3) : "—";

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold tracking-tight md:text-3xl">
            Experiment <span className="text-gradient">Overview</span>
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Live performance across the saved active-learning and RL strategies.
          </p>
        </div>
        <div className="flex gap-2">
          <Link to="/experiment-setup" className="inline-flex h-9 items-center rounded-md border border-border/60 px-4 text-sm font-medium hover:bg-accent">
            <FlaskConical className="mr-1.5 h-4 w-4" /> Setup
          </Link>
          <Link to="/annotation-lab">
            <Button className="bg-gradient-primary text-primary-foreground shadow-elegant">
              <Play className="mr-1.5 h-4 w-4" /> Open Simulation
            </Button>
          </Link>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <MetricCard label="AUC Score" value={last ? auc : selectedResult ? selectedResult.final_test_auc.toFixed(3) : "N/A"} hint={last ? "latest step" : "notebook test"} icon={<Gauge className="h-5 w-5" />} />
        <MetricCard label="Remaining Budget" value={`${budgetRemaining}`} hint={`of ${budget} queries`} tone="warning" icon={<Coins className="h-5 w-5" />} />
        <MetricCard label="Queries Used" value={`${used}`} hint="this run" icon={<Activity className="h-5 w-5" />} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <ChartContainer
          className="lg:col-span-2"
          title="Learning Efficiency Curve"
          description="AUC achieved per number of expert annotations queried"
          actions={
            <div className="flex flex-wrap items-center gap-1.5">
              {strategyDefinitions.map((definition) => {
                const s = definition.id;
                return (
                <Toggle
                  key={s}
                  size="sm"
                  pressed={visibleStrategies[s]}
                  onPressedChange={() => toggleStrategy(s)}
                  className="h-7 gap-1.5 text-xs"
                >
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ background: `var(--color-chart-${(strategyDefinitions.findIndex((item) => item.id === s) % 8) + 1})` }}
                  />
                  {definition.shortLabel}
                </Toggle>
              )})}
            </div>
          }
        >
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={curveData} margin={{ left: -10, right: 8, top: 8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="queries" stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} domain={[0.5, 1]} />
              <Tooltip contentStyle={tooltipStyle} />
              {strategyDefinitions.map((definition, index) => {
                const color = `var(--color-chart-${(index % 8) + 1})`;
                if (!visibleStrategies[definition.id]) return null;
                return (
                  <Line
                    key={definition.id}
                    type="monotone"
                    dataKey={definition.id}
                    stroke={color}
                    strokeWidth={definition.kind === "rl" ? 3 : 2}
                    dot={definition.kind === "rl" ? { r: 2 } : false}
                    name={definition.label}
                  />
                );
              })}
              <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>

        <div className="space-y-4">
          <ChartContainer title="Budget Status" description="Remaining annotation budget">
            <BudgetBar used={used} total={budget} />
            <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
              <div className="rounded-lg bg-success/10 p-2 text-success">
                <div className="font-mono-num text-base font-semibold">{budgetRemaining}</div>
                <div>Left</div>
              </div>
              <div className="rounded-lg bg-warning/10 p-2 text-warning">
                <div className="font-mono-num text-base font-semibold">{used}</div>
                <div>Used</div>
              </div>
              <div className="rounded-lg bg-primary/10 p-2 text-primary">
                <div className="font-mono-num text-base font-semibold">
                  {budget > 0 ? Math.round((used / budget) * 100) : 0}%
                </div>
                <div>Spent</div>
              </div>
            </div>
          </ChartContainer>

          <ChartContainer title="Recent Activity" description="Last 10 queries">
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={history.length ? history.slice(-10) : curveData.slice(-10)}>
                <Line
                  type="monotone"
                  dataKey={history.length ? "auc" : strategy}
                  stroke="var(--color-chart-1)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Tooltip contentStyle={tooltipStyle} />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      </div>
    </motion.div>
  );
}
